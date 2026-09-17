"""
    SciMLBase.postamble!(integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK})

ProbNumDiffEq.jl-specific implementation of SciMLBase's `postamble!` hook.

In addition to calling `OrdinaryDiffEqCore._postamble!(integ)`, calibrate the diffusion and
smooth the solution.
"""
function SciMLBase.postamble!(
    integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK},
)
    # OrdinaryDiffEqCore.jl-related calls:
    OrdinaryDiffEqCore._postamble!(integ)
    copyat_or_push!(integ.sol.k, integ.saveiter_dense, integ.k)
    pn_solution_endpoint_match_cur_integrator!(integ)

    # Calibrate the solution (if applicable)
    if isstatic(integ.cache.diffusionmodel)
        if integ.cache.diffusionmodel.calibrate
            # The estimated global_diffusion is just a scaling factor
            mle_diffusion = integ.cache.global_diffusion
            calibrate_solution!(integ, mle_diffusion)
        else
            constant_diffusion = integ.cache.default_diffusion
            set_diffusions!(integ.sol, constant_diffusion)
        end
    end
    if integ.alg.smooth
        smooth_solution!(integ)
    end

    @assert (length(integ.sol.u) == length(integ.sol.pu) == length(integ.sol.x_filt))

    return nothing
end

"""
    calibrate_solution!(integ, mle_diffusion)

Calibrate the solution (`integ.sol`) with the specified `mle_diffusion` by (i) setting the
values in `integ.sol.diffusions` to the `mle_diffusion` (see [`set_diffusions!`](@ref)),
(ii) rescaling all filtering estimates such that they have the correct diffusion, and (iii)
updating the solution estimates in `integ.sol.pu`.
"""
function calibrate_solution!(integ, mle_diffusion)

    # Set all solution diffusions; don't forget the initial diffusion!
    set_diffusions!(integ.sol, mle_diffusion * integ.cache.default_diffusion)

    # Rescale all filtering estimates to have the correct diffusion
    @assert mle_diffusion isa Diagonal
    @simd ivdep for C in integ.sol.x_filt.Σ
        apply_diffusion!(C, mle_diffusion)
    end
    @simd ivdep for C in integ.sol.backward_kernels.C
        apply_diffusion!(C, mle_diffusion)
    end

    # Keep the pre-stored smoother-state measurement quantities consistent with the
    # recalibrated covariances: the Kalman gains `K = Σ_pred Hᵀ S⁻¹` are invariant under
    # the rescaling (the measurement Jacobians of EK0 and DiagonalEK1 are dimension-pure,
    # and calibration with observation noise is ruled out by `ekargcheck`), while the
    # measurement covariances `S = H Σ_pred Hᵀ` pick up the same per-dimension congruence
    # as `Σ_pred` - so their Cholesky factors scale by the matching square root.
    for ss in integ.sol.smoother_states
        _rescale_measurement_chol!(ss.S_U, mle_diffusion)
    end

    # Re-write into the solution estimates
    for i in eachindex(integ.sol.pu, integ.sol.x_filt)
        _gaussian_mul!(integ.sol.pu[i], integ.cache.SolProj, integ.sol.x_filt[i])
    end
    # [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
end

"""
    set_diffusions!(solution::AbstractProbODESolution, diffusion::Union{Number,Diagonal})

Set the contents of `solution.diffusions` to the provided `diffusion`, overwriting the local
diffusion estimates that are in there. Typically, `diffusion` is either a global quasi-MLE
or the specified initial diffusion value if no calibration is desired.
"""
function set_diffusions!(solution::AbstractProbODESolution, diffusion)
    if diffusion isa Diagonal{<:Number,<:FillArrays.Fill}
        @simd ivdep for i in eachindex(solution.diffusions)
            solution.diffusions[i] = copy(diffusion)
        end
    elseif diffusion isa Diagonal{<:Number,<:Vector}
        @simd ivdep for d in solution.diffusions
            copy!(d, diffusion)
        end
    else
        throw(ArgumentError("unexpected diffusion type $(typeof(diffusion))"))
    end
    return nothing
end

"""
    smooth_solution!(integ)

Smooth the solution saved in `integ.sol`, filling `integ.sol.x_smooth` and updating the
values saved in `integ.sol.pu` and `integ.sol.u`.

Two implementations are available, selected by the solver's `smoother` argument:

- `:mbf` (default): the square-root Modified Bryson--Frazier (√MBF) smoother (Gibbs 2011),
  which propagates adjoint variables backward and only inverts the `d×d` measurement
  covariance (not the full `D×D` predicted state covariance as in the RTS smoother). The
  λ/Λ update and predict recursion is done in sqrt form in each step's local preconditioned
  coordinates for numerical stability; covariance recovery uses a hyperbolic QR
  factorization to guarantee a positive semi-definite result. The per-step measurement
  quantities are pre-stored by the forward pass in [`SmootherState`](@ref).
- `:rts`: the classic Rauch--Tung--Striebel smoother, marginalizing the backward transition
  kernels that were computed (in preconditioned square-root form) by the forward pass.

The actual per-step work is done by [`_mbf_backward_step!`](@ref) (MBF) or
`marginalize!` (RTT), both dispatching on the structure of the state covariances (dense,
EK0's Kronecker, or DiagonalEK1's block-diagonal), the same way [`predict_cov!`](@ref) and
[`update!`](@ref) do -- so the loops don't need to know or care which algorithm produced
the solution being smoothed.
"""
function smooth_solution!(integ)
    @unpack cache, sol = integ
    for (i, x) in enumerate(sol.x_filt)
        copyat_or_push!(sol.x_smooth, i, x)
    end

    if integ.alg.smoother == :mbf
        _smooth_solution_mbf!(integ)
    else
        _smooth_solution_rtt!(integ)
    end
    return nothing
end

"MBF branch of [`smooth_solution!`](@ref); see there."
function _smooth_solution_mbf!(integ)
    @unpack cache, sol = integ
    @unpack x_smooth, t, smoother_states = sol
    n = length(x_smooth)

    λ = zeros(eltype(x_smooth[1].μ), length(x_smooth[1].μ))
    # U_Λ needs to hold a general (non-diagonal) matrix once measurement info accumulates,
    # so it must be zero-initialized from the state covariance's structure (dense/Kronecker/
    # block-diagonal), not from `cache.P`/`PI` -- those are diagonal preconditioners even in
    # the dense (EK1) case, and a `Diagonal`-typed `U_Λ` can't hold the general update result.
    U_Λ = zero(x_smooth[1].Σ.R)

    # The dense (EK1) backward step works on full D×D matrices; preallocate its scratch
    # buffers once and reuse them across all steps so that the backward recursion runs
    # without per-step allocations. The structured (Kronecker / block-diagonal) paths only
    # ever touch small (q+1)-sized blocks, so they don't need this.
    scratch =
        U_Λ isa Matrix ? _MBFDenseScratch(length(λ), size(cache.H, 1), eltype(λ)) : nothing

    # x_smooth[n] = x_filt[n] exactly: there are no future measurements to smooth with.
    _gaussian_mul!(sol.pu[n], cache.SolProj, x_smooth[n])
    sol.u[n][:] .= sol.pu[n].μ

    for i in n:-1:2
        if iszero(t[i] - t[i-1])
            copy!(x_smooth[i-1], x_smooth[i])
            _gaussian_mul!(sol.pu[i-1], cache.SolProj, x_smooth[i-1])
            sol.u[i-1][:] .= sol.pu[i-1].μ
            continue
        end

        ss = smoother_states[i-1]
        dt_step = t[i] - t[i-1]
        make_transition_matrices!(cache, cache.prior, dt_step)

        λ, U_Λ = _mbf_backward_step!(
            x_smooth[i-1], λ, U_Λ, sol.x_filt[i-1], ss,
            cache.P, cache.PI, cache.A, scratch)

        _gaussian_mul!(sol.pu[i-1], cache.SolProj, x_smooth[i-1])
        sol.u[i-1][:] .= sol.pu[i-1].μ
    end
    return nothing
end

"RTT branch of [`smooth_solution!`](@ref); see there."
function _smooth_solution_rtt!(integ)
    @unpack cache, sol = integ
    @unpack x_smooth, t, backward_kernels = sol
    @unpack C_DxD, C_3DxD = cache

    @assert length(x_smooth) == length(backward_kernels) + 1

    for i in (length(x_smooth)-1):-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x_smooth[i], x_smooth[i+1])
        else
            marginalize!(x_smooth[i], x_smooth[i+1], backward_kernels[i]; C_DxD, C_3DxD)
        end
        _gaussian_mul!(sol.pu[i], cache.SolProj, x_smooth[i])
        sol.u[i][:] .= sol.pu[i].μ
    end
    return nothing
end

# Preallocated scratch buffers for the dense (EK1) backward step, created once per
# `smooth_solution!` call and reused by every step; sized by (D, d) = (state dimension,
# measurement dimension).
struct _MBFDenseScratch{T}
    U_Λ_prec::Matrix{T}
    U_Λ_pred::Matrix{T}
    U_Λ_new::Matrix{T}
    Stack::Matrix{T}
    R_M::Matrix{T}
    R_f::Matrix{T}
    U_f::Matrix{T}
    U_ΛK::Matrix{T}
    H_prec::Matrix{T}
    K_prec::Matrix{T}
    λ_prec::Vector{T}
    λ_pred::Vector{T}
    λ_new::Vector{T}
    w::Vector{T}
    z_sol::Vector{T}
    v_h::Vector{T}
    w_h::Vector{T}
    tau::Vector{T}
    S_U_T::Matrix{T}
end

function _MBFDenseScratch(D::Integer, d::Integer, ::Type{T}) where {T}
    return _MBFDenseScratch{T}(
        Matrix{T}(undef, D, D), Matrix{T}(undef, D, D), Matrix{T}(undef, D, D),
        Matrix{T}(undef, D + d, D),
        Matrix{T}(undef, D, D), Matrix{T}(undef, D, D),
        Matrix{T}(undef, D, D),
        Matrix{T}(undef, D, d), Matrix{T}(undef, d, D), Matrix{T}(undef, D, d),
        zeros(T, D), zeros(T, D), zeros(T, D),
        zeros(T, d), zeros(T, d),
        zeros(T, D), zeros(T, D), zeros(T, D),
        Matrix{T}(undef, d, d),
    )
end

"""
    _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A, scratch=nothing)

One backward step of the √MBF recursion: incorporate the measurement info in `ss` (or just
predict, if `ss === nothing`) into the adjoint state `(λ, U_Λ)`, then recover the smoothed
mean/covariance into `x_smooth_prev` using the filtered state `x_filt_prev`. Returns the
updated `(λ, U_Λ)` for use at the next (earlier) step.

The measurement quantities (`z`, `H`, `K`, `S_U`) were pre-computed and stored by the
forward pass in `ss` (see [`SmootherState`](@ref)), so this function does not need to
recompute the predicted covariance or the Kalman gain.

Dispatches on the structure of `P` (which matches that of the state covariances) so that EK0's
and DiagonalEK1's efficient Kronecker/block-diagonal representations are preserved: the λ/Λ
recursion and covariance recovery only ever touch the small `(q+1)×(q+1)` blocks (batched over
the `d` ODE dimensions for EK0, or looped independently for DiagonalEK1), never a dense `D×D`
matrix -- without this, backward smoothing would cost `O(D^3) = O((d(q+1))^3)` instead of the
`O(d)`-ish cost the rest of the package achieves for these algorithms.

For the dense (EK1) case, `scratch` (a [`_MBFDenseScratch`](@ref); built on the fly if not
provided) holds preallocated buffers so that the `D×D` work runs without per-step
allocations. The structured methods ignore it.

The λ/Λ update and predict steps involve matrix products (H, K, the transition matrix) applied
repeatedly across many steps; doing this in physical coordinates re-introduces the same
ill-conditioning that motivated preconditioning elsewhere in this codebase (IWP covariances
span many orders of magnitude across derivative orders). So: precondition into this specific
transition's local coordinates (`P`, `PI` depend on the step size and are only valid for this
one transition), do the update+predict there using the h-independent preconditioned transition
`A`, then unprecondition back to physical before recovering `x_smooth_prev` below (using this
transition's own `P`/`PI`, the natural local scale for `x_filt_prev` itself -- a scale borrowed
from an adjacent transition was found empirically to reintroduce significant error).
"""
function _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A)
    D = length(λ)
    d = isnothing(ss) ? D : size(ss.H, 1)
    return _mbf_backward_step!(
        x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A,
        _MBFDenseScratch(D, d, eltype(λ)),
    )
end

# `smooth_solution!` passes its (possibly `nothing`) scratch through; the structured
# (Kronecker / block-diagonal) methods below ignore it and dispatch on `P`'s structure.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A, scratch,
)
    return _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A)
end

# The dense (EK1) implementation, allocation-free given a matching `scratch`.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ, x_filt_prev, ss,
    P::Diagonal, PI::Diagonal, A::Matrix, sc::_MBFDenseScratch,
)
    D = length(λ)
    StackTop = view(sc.Stack, 1:D, :)

    if isnothing(ss)
        copyto!(sc.λ_prec, λ)
        mul!(sc.λ_new, transpose(A), sc.λ_prec)
        copyto!(sc.U_Λ_prec, U_Λ)
        mul!(sc.U_Λ_new, sc.U_Λ_prec, A)
    else
        d = size(sc.H_prec, 1)
        StackBot = view(sc.Stack, (D+1):(D+d), :)

        # Precondition into this transition's local coordinates.
        mul!(sc.H_prec, ss.H, transpose(PI))
        mul!(sc.K_prec, P, ss.K)
        mul!(sc.λ_prec, transpose(PI), λ)
        mul!(sc.U_Λ_prec, U_Λ, PI)

        _sqrt_mbf_update!(
            sc.λ_prec, sc.U_Λ_prec, sc.K_prec, ss.S_U, ss.z, sc.H_prec, D, sc,
        )

        # Predict (still in preconditioned coordinates) and unprecondition.
        mul!(sc.λ_pred, transpose(A), sc.λ_prec)
        mul!(sc.λ_new, transpose(P), sc.λ_pred)
        mul!(sc.U_Λ_pred, sc.U_Λ_prec, A)
        mul!(sc.U_Λ_new, sc.U_Λ_pred, P)
    end
    λ_new, U_Λ_new = sc.λ_new, sc.U_Λ_new

    # Recover smoothed mean/covariance using the now-updated λ, Λ (information from
    # measurements {i, ..., n}).
    copy!(sc.U_f, x_filt_prev.Σ.R)
    mul!(sc.λ_prec, sc.U_f, λ_new)
    mul!(sc.λ_pred, transpose(sc.U_f), sc.λ_prec)
    x_smooth_prev.μ .-= sc.λ_pred
    # The recovery is done directly in physical coordinates: `V = U_Λ_new * U_f'` and
    # `R_f = R_M * U_f` (no P/PI scalings -- they would cancel exactly, since
    # `PI == P^{-1}`). This saves two D×D products per step.
    mul!(sc.U_Λ_pred, U_Λ_new, transpose(sc.U_f))
    _hyperbolic_qr!(sc.R_M, sc.v_h, sc.w_h, sc.U_Λ_pred)
    mul!(sc.R_f, sc.R_M, sc.U_f)
    # `sc.R_f` is a valid square-root factor of the smoothed covariance
    # (`sc.R_f'sc.R_f = Σ_s`) but generally a full (non-triangular) matrix. That is fine:
    # the package does not assume upper-triangular factors in general (e.g. `update!`
    # stores `Σ_pred.R * (I - K*H)'`), and all `PSDMatrix` operations (`Matrix`, `det`,
    # `logabsdet`, `diag`, QR stacks, ...) only rely on `R'R`. So we store the factor
    # as-is instead of paying for a per-step re-triangularizing QR.
    copy!(x_smooth_prev.Σ.R, sc.R_f)

    return λ_new, U_Λ_new
end

# Kronecker version (EK0): reduce to the shared small `.B` block, batched over the `d` ODE
# dimensions via `(q+1)×d`-shaped λ, instead of densifying to `D×D`.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ::IsometricKroneckerProduct, x_filt_prev, ss,
    P::IsometricKroneckerProduct, PI::IsometricKroneckerProduct,
    A::IsometricKroneckerProduct,
)
    d = P.rdim
    Q = size(P.B, 1)
    _λ = reshape_no_alloc(λ, d, Q)'
    P_B, PI_B, A_B, U_Λ_B = P.B, PI.B, A.B, U_Λ.B

    if isnothing(ss)
        λ_new_B = A_B' * _λ
        U_Λ_new_B = U_Λ_B * A_B
    else
        K_B = ss.K.B
        S_U_B = ss.S_U.B

        H_prec = ss.H.B * PI_B
        K_prec = P_B * K_B
        λ_prec = PI_B' * _λ
        U_Λ_prec = U_Λ_B * PI_B
        z_row = reshape_no_alloc(ss.z, d, 1)'

        _sqrt_mbf_update!(λ_prec, U_Λ_prec, K_prec, S_U_B, z_row, H_prec, Q)

        λ_prec = A_B' * λ_prec
        U_Λ_prec = U_Λ_prec * A_B

        λ_new_B = P_B' * λ_prec
        U_Λ_new_B = U_Λ_prec * P_B
    end

    U_f_B = x_filt_prev.Σ.R.B
    μ_smooth_view = reshape_no_alloc(x_smooth_prev.μ, d, Q)'
    _mbf_recover_mean!(μ_smooth_view, U_f_B, λ_new_B)
    # Same as above: recover directly in physical coordinates -- the P/PI scalings cancel
    # exactly, so `U_Λ_B * PI_B` and `U_f_B * P_B'` (and the trailing `* PI_B'`) are not needed.
    new_R_B = _mbf_recover_cov(x_filt_prev.Σ.R.B, U_Λ_new_B)
    copy!(x_smooth_prev.Σ.R.B, new_R_B)

    λ_new = similar(λ)
    reshape_no_alloc(λ_new, d, Q)' .= λ_new_B
    return λ_new, IsometricKroneckerProduct(d, U_Λ_new_B)
end

# Block-diagonal version (DiagonalEK1): unlike EK0, each of the `d` ODE dimensions has its own
# measurement Jacobian (and hence its own Kalman gain), so there is no shared small block to
# batch over. Instead, reduce to `d` independent `(q+1)`-sized problems, one per diagonal
# block, and recurse -- each recursive call dispatches to the dense method above.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ::BlocksOfDiagonals, x_filt_prev, ss,
    P::BlocksOfDiagonals, PI::BlocksOfDiagonals, A::BlocksOfDiagonals,
)
    d = length(blocks(P))
    D = length(λ)
    λ_new = similar(λ)
    new_U_Λ_blocks = similar(blocks(U_Λ))

    for bi in 1:d
        _ss =
            isnothing(ss) ? nothing :
            SmootherState(
                collect(view(ss.z, bi:bi)), ss.H.blocks[bi],
                ss.K.blocks[bi], ss.S_U.blocks[bi],
            )
        _x_filt_prev =
            Gaussian(view(x_filt_prev.μ, bi:d:D), PSDMatrix(x_filt_prev.Σ.R.blocks[bi]))
        _x_smooth_prev =
            Gaussian(view(x_smooth_prev.μ, bi:d:D), PSDMatrix(x_smooth_prev.Σ.R.blocks[bi]))

        λ_bi_new, U_Λ_bi_new = _mbf_backward_step!(
            _x_smooth_prev, collect(view(λ, bi:d:D)), U_Λ.blocks[bi],
            _x_filt_prev, _ss, P.blocks[bi], PI.blocks[bi], A.blocks[bi])

        λ_new[bi:d:D] .= λ_bi_new
        new_U_Λ_blocks[bi] = U_Λ_bi_new
    end
    return λ_new, BlocksOfDiagonals(new_U_Λ_blocks)
end

function _mbf_recover_mean!(μ_smooth, U_f::Matrix, λ)
    μ_smooth .-= U_f' * (U_f * λ)
end

"""
    _mbf_recover_cov(U_f, U_Λ)

Recover the smoothed covariance square-root factor from the filtered covariance
(`U_f`, upper triangular such that `Σ_filt = U_f'U_f`) and the MBF adjoint
information matrix (`U_Λ`, such that `Λ = U_Λ'U_Λ`), i.e. `R` with
`R'R = Σ_filt - Σ_filt*Λ*Σ_filt`. Writing `Σ_smooth = U_f' * (I - W*W') * U_f` with
`W = U_f*U_Λ'`, this is computed via a hyperbolic QR factorization of `V = W'`, which
guarantees a PSD result (unlike forming `Σ_filt - Σ_filt*Λ*Σ_filt` densely, which
squares the condition number and can catastrophically cancel when the smoothed
covariance is much smaller than the filtered one). `U_f` is a factor of the filtered
covariance and `U_Λ` of the adjoint information matrix, both in the same coordinates.
"""
function _mbf_recover_cov(U_f::Matrix, U_Λ::Matrix)
    V = U_Λ * U_f'
    R_M = _hyperbolic_qr!(V)
    return R_M * U_f
end

function _hyperbolic_qr!(V::Matrix{T}) where {T}
    d = size(V, 1)
    return _hyperbolic_qr!(
        Matrix{T}(I, d, d), Vector{T}(undef, d), Vector{T}(undef, d), V,
    )
end

# In-place variant of [`_hyperbolic_qr!`](@ref) using the preallocated output matrix `top`
# (initialized to I here) and work vectors `v_h`, `w_h`; destroys `V`.
function _hyperbolic_qr!(
    top::Matrix{T}, v_h::Vector{T}, w_h::Vector{T}, V::Matrix{T},
) where {T}
    d = size(V, 1)
    fill!(top, zero(T))
    @inbounds for i in 1:d
        top[i, i] = one(T)
    end
    bot = V
    @inbounds for i in 1:d
        na = d - i + 1

        asq = zero(T)
        for l in i:d
            asq += top[l, i] * top[l, i]
        end
        bsq = zero(T)
        for l in 1:d
            bsq += bot[l, i] * bot[l, i]
        end

        α² = max(asq - bsq, zero(T))
        α = sqrt(α²)

        if α < eps(T) * d
            for l in i:d
                top[l, i] = zero(T)
            end
            for l in 1:d
                bot[l, i] = zero(T)
            end
            continue
        end

        α = top[i, i] >= 0 ? -α : α
        β = one(T) / (α * (α - top[i, i]))

        v_h[1] = top[i, i] - α
        for l in 2:na
            v_h[l] = top[i+l-1, i]
        end
        for l in 1:d
            w_h[l] = bot[l, i]
        end

        for j in (i+1):d
            τ = zero(T)
            for l in 1:na
                τ += v_h[l] * top[i+l-1, j]
            end
            for l in 1:d
                τ -= w_h[l] * bot[l, j]
            end
            τ *= β

            for l in 1:na
                top[i+l-1, j] -= τ * v_h[l]
            end
            for l in 1:d
                bot[l, j] -= τ * w_h[l]
            end
        end

        top[i, i] = α
        for l in (i+1):d
            top[l, i] = zero(T)
        end
        for l in 1:d
            bot[l, i] = zero(T)
        end
    end
    return top
end

"""
    _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D[, sc])

Information-form update of the MBF adjoint state `(λ, U_Λ)` with the measurement quantities
of one step (`K`, `S_U`, `z`, `H`).

λ update (BK-free form): with `z_code = h(m_pred)` and innovation `-z_code`,
`λ ← λ - H' (K'λ - S⁻¹ z_code)`.

Λ update (sqrt form via QR of the stack `[U_Λ * (I - K*H); M]`): the second stack block must
be a factor `M` with `M'M = S⁻¹`. With the upper triangular Cholesky factor `S_U`
(`S = S_U'S_U`) that is `M = S_U⁻ᵀ H`, since `(S_U⁻ᵀ H)'(S_U⁻ᵀ H) = H'S⁻¹H` (the lower
triangular solve with `S_U'` turns `S_U` into an inverted *lower* triangular factor);
instead using `M = S_U⁻¹ H` would give `H'(S_U S_U')⁻¹H ≠ H'S⁻¹H` (unless `S_U` is
diagonal) and silently corrupt Λ.
"""
function _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D)
    S_chol = Cholesky(S_U, 'U', 0)

    # λ update (BK-free form); see the docstring for the sign conventions
    w = K' * λ - S_chol \ z
    λ .-= H' * w

    # Λ update (sqrt form via QR of the stack; see the docstring for the S_U⁻ᵀH factor).
    BK = -K * H
    @inbounds for j in 1:D
        BK[j, j] += 1
    end
    Stack = [U_Λ * BK; S_chol.U' \ H]
    U_Λ .= _positive_qr_r(Stack)
end

# Dense version writing into the preallocated buffers of `sc` (see `_MBFDenseScratch`),
# so that the backward recursion runs without per-step allocations. Mathematically identical
# to the method above (used by the structured paths, which only touch small blocks).
function _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D, sc::_MBFDenseScratch)
    # `S_U_T = S_U'` as a contiguous matrix so that the triangular solves below hit BLAS.
    copyto!(sc.S_U_T, transpose(S_U))

    # λ update (BK-free form); see the docstring for the sign conventions
    mul!(sc.w, transpose(K), λ)
    copyto!(sc.z_sol, z)
    ldiv!(LowerTriangular(sc.S_U_T), sc.z_sol)
    ldiv!(UpperTriangular(S_U), sc.z_sol)
    sc.w .-= sc.z_sol
    mul!(λ, transpose(H), sc.w, -1.0, 1.0)

    # Λ update (sqrt form via QR of the stack; see the docstring for the S_U⁻ᵀH factor).
    # The first stack row `U_Λ * (I - K*H)` is computed as `U_Λ - (U_Λ*K)*H` to avoid
    # materializing the D×D matrix `BK`.
    mul!(sc.U_ΛK, U_Λ, K)
    StackTop = view(sc.Stack, 1:D, :)
    copyto!(StackTop, U_Λ)
    mul!(StackTop, sc.U_ΛK, H, -1.0, 1.0)
    StackBot = view(sc.Stack, (D+1):(D+size(H, 1)), :)
    copyto!(StackBot, H)
    ldiv!(LowerTriangular(sc.S_U_T), StackBot)
    _positive_qr_r!(U_Λ, sc.Stack, sc.tau)
    return nothing
end

"""
    _save_smoother_state!(smoother_states, cache)

Store the measurement quantities that the √MBF backward smoother needs for this step's
update: the measurement mean `z = h(x_pred)` (left in `cache.measurement.μ`; the filter's
innovation is `0 - z`, since the update assumes zero measurements), the measurement Jacobian
`H`, the Kalman gain `K` (left in `cache.C_Dxd` by `update!`), and the upper triangular
Cholesky factor `S_U` of the measurement covariance (plus `R` if observation noise is
configured).

`K` and `S_U` are exactly the quantities the forward filter's update used, so backward
smoothing with them reproduces the filter's posterior exactly; see
[`calibrate_solution!`](@ref) for how they stay consistent when the solution is
calibrated after the solve.
"""
function _save_smoother_state!(smoother_states, cache)
    S_U = _measurement_cholesky(cache.measurement.Σ)
    if S_U === nothing
        # Degenerate step: the predicted covariance was (numerically) zero, so `update!`
        # skipped the update. There is no measurement information to smooth with, so the
        # backward pass at this step does a plain prediction (`ss === nothing` branch of
        # `_mbf_backward_step!`).
        push!(smoother_states, nothing)
        return nothing
    end
    T = eltype(cache.C_Dxd)
    H = copy(cache.H)
    z = Vector{T}(cache.measurement.μ)
    K = copy(cache.C_Dxd)
    push!(smoother_states, SmootherState(z, H, K, S_U))
    return nothing
end

"""
    _measurement_cholesky(S)

Upper triangular Cholesky factor of the measurement covariance `S`, stored in the same
matrix structure as `S` itself (dense `Matrix`, `IsometricKroneckerProduct`, or
`BlocksOfDiagonals`), so that the backward smoother can dispatch on it like on `H` and
`K`.

If `S` is exactly zero (which means the predicted covariance was (numerically) zero too, so
`update!` skipped the update) returns `nothing`: there is no measurement information to
smooth with, and `_save_smoother_state!` stores a `nothing` smoother state for that step.
Throws if `S` is not factorizable for any other reason, since `update!` has already
factorized the same matrix by the time this is called.
"""
function _measurement_cholesky(S::Matrix{T}) where {T}
    F = cholesky(Symmetric(S); check=false)
    issuccess(F) && return Matrix(F.U)
    # An exactly zero measurement covariance means the predicted covariance was (numer-
    # ically) zero too, so `update!` skipped the update and no measurement information is
    # available to smooth with. Signal that to `_save_smoother_state!`.
    iszero(S) && return nothing
    # `update!` has already factorized this exact matrix successfully by the time the
    # smoother state is saved, so a failure here can only indicate a bug elsewhere - fail
    # loudly instead of silently corrupting the Λ update with some made-up factor.
    error(
        "Cholesky factorization of the measurement covariance failed; " *
        "cannot store the smoother state for backward smoothing.")
end
_measurement_cholesky(S::IsometricKroneckerProduct) = begin
    S_U = _measurement_cholesky(S.B)
    S_U === nothing ? nothing : IsometricKroneckerProduct(S.rdim, S_U)
end
function _measurement_cholesky(S::BlocksOfDiagonals)
    chol_blocks = [_measurement_cholesky(block) for block in blocks(S)]
    # A fully degenerate step (all blocks zero) signals "no measurement info"; the mixed case
    # is unreachable in practice (`update!` would already have failed on the zero block).
    all(isnothing, chol_blocks) && return nothing
    any(isnothing, chol_blocks) && error(
        "Partially degenerate measurement covariance; cannot store a smoother state.")
    return BlocksOfDiagonals(chol_blocks)
end

"""
    _rescale_measurement_chol!(S_U, mle_diffusion)

Rescale a stored smoother-state measurement-covariance Cholesky factor `S_U` after the
solution's covariances were calibrated with `mle_diffusion` (see
[`calibrate_solution!`](@ref)). The calibrated model's measurement covariance is
`S' = M S M` with `M = sqrt.(mle_diffusion)` (a uniform scaling for isotropic
diffusions, a per-dimension one for MV diffusions -- where the measurement Jacobians of
EK0 and DiagonalEK1 are dimension-pure), so its Cholesky factor is `S_U' = S_U M`.
"""
function _rescale_measurement_chol!(S_U, mle_diffusion::Diagonal)
    if mle_diffusion isa Diagonal{<:Number,<:FillArrays.Fill}
        _rescale_measurement_chol_uniform!(S_U, sqrt(mle_diffusion.diag.value))
    else
        _rescale_measurement_chol_perdim!(S_U, Diagonal(sqrt.(mle_diffusion.diag)))
    end
end
function _rescale_measurement_chol_uniform!(S_U::Matrix, σ::Number)
    rmul!(S_U, σ)
    return S_U
end
function _rescale_measurement_chol_uniform!(S_U::IsometricKroneckerProduct, σ::Number)
    rmul!(S_U.B, σ)
    return S_U
end
function _rescale_measurement_chol_uniform!(S_U::BlocksOfDiagonals, σ::Number)
    @simd ivdep for i in eachindex(blocks(S_U))
        rmul!(blocks(S_U)[i], σ)
    end
    return S_U
end
function _rescale_measurement_chol_perdim!(S_U::Matrix, M::Diagonal)
    rmul!(S_U, M)
    return S_U
end
function _rescale_measurement_chol_perdim!(S_U::BlocksOfDiagonals, M::Diagonal)
    @simd ivdep for i in eachindex(blocks(S_U))
        rmul!(blocks(S_U)[i], M.diag[i])
    end
    return S_U
end
function _rescale_measurement_chol_perdim!(S_U::IsometricKroneckerProduct, M::Diagonal)
    # Only reachable when `IsometricKroneckerCovariance` was constructed explicitly with a
    # MV diffusion model: `covariance_structure` would pick `IsometricKroneckerCovariance`
    # automatically only for scalar diffusions. The isometric representation keeps the
    # per-dimension scaling in the `Σ_d` block shared across ODE dimensions, so it cannot
    # represent the per-dimension rescale `S_U' = S_U * M` - fail loudly instead.
    throw(
        ArgumentError(
            "Per-dimension diffusion calibration is not supported with isometric Kronecker covariances.",
        ),
    )
end

function _positive_qr_r(Stack)
    R = Matrix(qr(Stack).R)
    @inbounds for i in axes(R, 1)
        if R[i, i] < 0
            @views R[i, :] .*= -1
        end
    end
    return R
end

# In-place variant of [`_positive_qr_r`](@ref): factorizes `Stack` (destroying it) and writes
# the R factor with positive diagonal into `R_dest` (zeros below the diagonal).
function _positive_qr_r!(R_dest::Matrix, Stack::Matrix, tau::Vector)
    D = size(Stack, 2)
    if eltype(Stack) <: LinearAlgebra.BlasFloat
        LinearAlgebra.LAPACK.geqrf!(Stack, tau)
    else
        # Generic fallback for e.g. ForwardDiff.Dual eltypes (allocating, but AD rarely
        # smooths large systems).
        Stack[1:D, 1:D] .= qr(Stack).R
    end
    @inbounds for i in 1:D
        if Stack[i, i] < 0
            @views Stack[i, 1:D] .*= -1
        end
    end
    @inbounds for j in 1:D, i in 1:D
        R_dest[i, j] = i <= j ? Stack[i, j] : zero(eltype(R_dest))
    end
    return R_dest
end

"Inspired by `OrdinaryDiffEqCore.solution_match_cur_integrator!`"
function pn_solution_endpoint_match_cur_integrator!(integ)
    if integ.opts.save_end
        i = integ.saveiter

        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)

        copyat_or_push!(
            integ.sol.pu,
            i,
            _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x),
        )

        if !integ.opts.save_everystep && i > 1
            save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
        end
    end
end

"""
    save_diffusion!(sol, i, diffusion)

`copyat_or_push!` for `sol.diffusions`, which needs to replace entries instead of copying
into them: the scalar diffusions are `Fill`-backed `Diagonal`s and cannot be mutated.
"""
function save_diffusion!(sol, i, diffusion)
    if i <= length(sol.diffusions)
        sol.diffusions[i] = copy(diffusion)
    else
        push!(sol.diffusions, copy(diffusion))
    end
    return nothing
end

"Extends `OrdinaryDiffEqCore._savevalues!` to save ProbNumDiffEq.jl-specific things."
function DiffEqBase.savevalues!(
    integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK},
    force_save=false,
    reduce_size=true,
)

    # Do whatever OrdinaryDiffEqCore would do
    out = OrdinaryDiffEqCore._savevalues!(integ, force_save, reduce_size)

    # Save our custom stuff that we need for the posterior
    if integ.opts.save_everystep
        i = integ.saveiter
        save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)
        _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x)
        copyat_or_push!(integ.sol.pu, i, integ.cache.pu_tmp)

        if integ.alg.smoother == :rts || integ.alg.save_backward_kernels
            copyat_or_push!(
                integ.sol.backward_kernels, i, integ.cache.backward_kernel)
        end
        if integ.alg.smooth && integ.alg.smoother == :mbf
            _save_smoother_state!(integ.sol.smoother_states, integ.cache)
        end
    end

    return out
end

function OrdinaryDiffEqCore.update_uprev!(
    integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK},
)
    @assert !OrdinaryDiffEqCore.alg_extrapolates(integ.alg)
    @assert isinplace(integ.sol.prob)
    @assert !(integ.alg isa OrdinaryDiffEqCore.DAEAlgorithm)

    recursivecopy!(integ.uprev, integ.u)
    recursivecopy!(integ.cache.xprev, integ.cache.x)
    nothing
end
