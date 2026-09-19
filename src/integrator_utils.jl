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
    mle_scale = _measurement_chol_scale(mle_diffusion)
    for ss in integ.sol.smoother_states
        ss === nothing && continue
        _rescale_measurement_chol!(ss.S_U, mle_scale)
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
`marginalize!` (RTS), both dispatching on the structure of the state covariances (dense,
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
        _smooth_solution_rts!(integ)
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

    # Preallocate scratch buffers once and reuse them across all backward steps.
    T = eltype(λ)
    scratch = if U_Λ isa Matrix
        _MBFDenseScratch(length(λ), size(cache.H, 1), T)
    elseif U_Λ isa IsometricKroneckerProduct
        Q = size(U_Λ.B, 1)
        _MBFDenseScratch(Q, size(cache.H.B, 1), U_Λ.rdim, T)
    elseif U_Λ isa BlocksOfDiagonals
        Q = size(blocks(U_Λ)[1], 1)
        _MBFDenseScratch(Q, size(blocks(cache.H)[1], 1), T)
    else
        error("unsupported state covariance type: $(typeof(U_Λ))")
    end

    # x_smooth[n] = x_filt[n] exactly: there are no future measurements to smooth with.
    _gaussian_mul!(sol.pu[n], cache.SolProj, x_smooth[n])
    sol.u[n][:] .= sol.pu[n].μ

    # The backward loop restores per-step rate parameters into the cache's prior (IOUP with
    # `update_rate_parameter=true`); leave the cache at the forward-final value afterwards
    # rather than the earliest step's (the loop's last restoration), matching what a
    # non-smoothed solve leaves behind, in case anything downstream inspects `cache.prior`
    # post-solve.
    saved_rate_parameter =
        (cache.prior isa IOUP && cache.prior.update_rate_parameter) ?
        copy(cache.prior.rate_parameter) : nothing

    for i in n:-1:2
        if iszero(t[i] - t[i-1])
            copy!(x_smooth[i-1], x_smooth[i])
            _gaussian_mul!(sol.pu[i-1], cache.SolProj, x_smooth[i-1])
            sol.u[i-1][:] .= sol.pu[i-1].μ
            continue
        end

        ss = smoother_states[i-1]
        dt_step = t[i] - t[i-1]
        # Restore the rate parameter of this step's transition (IOUP with
        # `update_rate_parameter=true`): `cache.prior` still holds the value of the last
        # forward step, but the transition matrices of this step were computed with this
        # step's value.
        if (cache.prior isa IOUP && cache.prior.update_rate_parameter) &&
           ss isa SmootherState
            copyto!(cache.prior.rate_parameter, ss.rate_parameter)
        end
        make_transition_matrices!(cache, cache.prior, dt_step)

        λ, U_Λ = _mbf_backward_step!(
            x_smooth[i-1], λ, U_Λ, sol.x_filt[i-1], ss,
            cache.P, cache.PI, cache.A, scratch)

        _gaussian_mul!(sol.pu[i-1], cache.SolProj, x_smooth[i-1])
        sol.u[i-1][:] .= sol.pu[i-1].μ
    end

    if !isnothing(saved_rate_parameter)
        copyto!(cache.prior.rate_parameter, saved_rate_parameter)
    end

    return nothing
end

"RTS branch of [`smooth_solution!`](@ref); see there."
function _smooth_solution_rts!(integ)
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
struct _MBFDenseScratch{T,Tw<:AbstractVecOrMat{T}}
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
    w::Tw
    z_sol::Tw
    v_h::Vector{T}
    w_h::Vector{T}
    tau::Vector{T}
    S_U_T::Matrix{T}
end

function _MBFDenseScratch(D::Integer, d::Integer, ::Type{T}) where {T}
    return _MBFDenseScratch(
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

function _MBFDenseScratch(D::Integer, d::Integer, d_ode::Integer, ::Type{T}) where {T}
    return _MBFDenseScratch(
        Matrix{T}(undef, D, D), Matrix{T}(undef, D, D), Matrix{T}(undef, D, D),
        Matrix{T}(undef, D + d, D),
        Matrix{T}(undef, D, D), Matrix{T}(undef, D, D),
        Matrix{T}(undef, D, D),
        Matrix{T}(undef, D, d), Matrix{T}(undef, d, D), Matrix{T}(undef, D, d),
        zeros(T, D), zeros(T, D), zeros(T, D),
        zeros(T, d, d_ode), zeros(T, d, d_ode),
        zeros(T, D), zeros(T, D), zeros(T, D),
        Matrix{T}(undef, d, d),
    )
end

"""
    _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A, sc)

One backward step of the √MBF recursion: incorporate the measurement info in `ss` (or just
predict, if `ss === nothing`) into the adjoint state `(λ, U_Λ)`, then recover the smoothed
mean/covariance into `x_smooth_prev` using the filtered state `x_filt_prev`. Returns the
updated `(λ, U_Λ)` for use at the next (earlier) step.

The measurement quantities (`z`, `H`, `K`, `S_U`) were pre-computed and stored by the
forward pass in `ss` (see [`SmootherState`](@ref)), so this function does not need to
recompute the predicted covariance or the Kalman gain.

**AD limitation**: under forward-mode AD (ForwardDiff.Dual eltypes), the smoothed
*values* computed here are correct, but the partials of the smoothed *covariance*
(`x_smooth_prev.Σ`) can be `NaN`; smoothed means are unaffected. This traces to the
hyperbolic QR factorization in [`_hyperbolic_qr!`](@ref) used by [`_mbf_recover_cov`](@ref):
its rotation is ~1/α-conditioned in the J-norm `α` of a column, and as `α → 0⁺` (a nearly
singular `I - V'V`, i.e. smoothed covariance much smaller than filtered) the partials of
`α = sqrt(max(asq - bsq, 0))` diverge before the degenerate-column threshold is reached,
which then cancels catastrophically downstream. This is an intrinsic conditioning property
of hyperbolic rotations (Gibbs 2011), not a bug in this implementation, and there is no
known fix that preserves the Float64 accuracy of the degenerate-column handling. Use
`smoother=:rts` if you need gradients through smoothed covariances.

Dispatches on the structure of `P` (which matches that of the state covariances) so that EK0's
and DiagonalEK1's efficient Kronecker/block-diagonal representations are preserved: the λ/Λ
recursion and covariance recovery only ever touch the small `(q+1)×(q+1)` blocks (batched over
the `d` ODE dimensions for EK0, or looped independently for DiagonalEK1), never a dense `D×D`
matrix -- without this, backward smoothing would cost `O(D^3) = O((d(q+1))^3)` instead of the
`O(d)`-ish cost the rest of the package achieves for these algorithms.

`sc` is a [`_MBFDenseScratch`](@ref) holding preallocated buffers, created once by
[`_smooth_solution_mbf!`](@ref) and reused across all backward steps.

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

# The dense (EK1) implementation, allocation-free given a matching `scratch`.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ, x_filt_prev, ss,
    P::Diagonal, PI::Diagonal, A::Matrix, sc::_MBFDenseScratch,
)
    D = length(λ)
    StackTop = view(sc.Stack, 1:D, :)

    if isnothing(ss)
        # Pure predict (no measurement): λ_new = Φ'λ, U_Λ_new = U_Λ Φ, where
        # the physical transition Φ = PI A P, so Φ' = P A' PI (diagonal P, PI).
        mul!(sc.λ_prec, PI, λ)
        mul!(sc.λ_pred, transpose(A), sc.λ_prec)
        mul!(sc.λ_new, P, sc.λ_pred)
        mul!(sc.U_Λ_prec, U_Λ, PI)
        mul!(sc.U_Λ_pred, sc.U_Λ_prec, A)
        mul!(sc.U_Λ_new, sc.U_Λ_pred, P)
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
    A::IsometricKroneckerProduct, sc::_MBFDenseScratch,
)
    d = P.rdim
    Q = size(P.B, 1)
    _λ = reshape_no_alloc(λ, d, Q)'
    P_B, PI_B, A_B, U_Λ_B = P.B, PI.B, A.B, U_Λ.B

    if isnothing(ss)
        λ_new_B = P_B' * (A_B' * (PI_B' * _λ))
        U_Λ_new_B = (U_Λ_B * PI_B) * A_B * P_B
    else
        K_B = ss.K.B
        S_U_B = ss.S_U.B

        H_prec = ss.H.B * PI_B
        K_prec = P_B * K_B
        λ_prec = PI_B' * _λ
        U_Λ_prec = U_Λ_B * PI_B
        z_row = reshape_no_alloc(ss.z, d, 1)'

        _sqrt_mbf_update!(λ_prec, U_Λ_prec, K_prec, S_U_B, z_row, H_prec, Q, sc)

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
    sc::_MBFDenseScratch,
)
    d = length(blocks(P))
    D = length(λ)
    λ_new = similar(λ)
    new_U_Λ_blocks = similar(blocks(U_Λ))

    for bi in 1:d
        _ss =
            isnothing(ss) ? nothing :
            SmootherState(
                view(ss.z, bi:bi), ss.H.blocks[bi],
                ss.K.blocks[bi], ss.S_U.blocks[bi], nothing,
            )
        _x_filt_prev =
            Gaussian(view(x_filt_prev.μ, bi:d:D), PSDMatrix(x_filt_prev.Σ.R.blocks[bi]))
        _x_smooth_prev =
            Gaussian(view(x_smooth_prev.μ, bi:d:D), PSDMatrix(x_smooth_prev.Σ.R.blocks[bi]))

        λ_bi_new, U_Λ_bi_new = _mbf_backward_step!(
            _x_smooth_prev, view(λ, bi:d:D), U_Λ.blocks[bi],
            _x_filt_prev, _ss, P.blocks[bi], PI.blocks[bi], A.blocks[bi], sc)

        λ_new[bi:d:D] .= λ_bi_new
        new_U_Λ_blocks[bi] = copy(U_Λ_bi_new)
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
#
# AD limitation: `β = 1 / (α * (α - top[i, i]))` below is ~1/α²-conditioned in the column's
# J-norm `α = sqrt(max(asq - bsq, 0))`. Under ForwardDiff.Dual eltypes, the *values* are
# unaffected, but as `α → 0⁺` (short of the `α < eps(T) * d` degenerate-column cutoff) the
# partials of `α` (and hence of `β`, `τ`) diverge, which can produce NaN partials in the
# recovered smoothed covariance further downstream (see `_mbf_backward_step!`'s docstring).
# This is intrinsic to the hyperbolic rotation's conditioning (Gibbs 2011), not fixable here
# without weakening the degenerate-column handling that keeps the Float64 values accurate.
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
    _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D, sc)

Information-form update of the MBF adjoint state `(λ, U_Λ)` with the measurement quantities
of one step (`K`, `S_U`, `z`, `H`), writing into the preallocated buffers of `sc`
(a [`_MBFDenseScratch`](@ref)).

λ update (BK-free form): with `z_code = h(m_pred)` and innovation `-z_code`,
`λ ← λ - H' (K'λ - S⁻¹ z_code)`.

Λ update (sqrt form via QR of the stack `[U_Λ * (I - K*H); M]`): the second stack block must
be a factor `M` with `M'M = S⁻¹`. With the upper triangular Cholesky factor `S_U`
(`S = S_U'S_U`) that is `M = S_U⁻ᵀ H`, since `(S_U⁻ᵀ H)'(S_U⁻ᵀ H) = H'S⁻¹H` (the lower
triangular solve with `S_U'` turns `S_U` into an inverted *lower* triangular factor);
instead using `M = S_U⁻¹ H` would give `H'(S_U S_U')⁻¹H ≠ H'S⁻¹H` (unless `S_U` is
diagonal) and silently corrupt Λ.
"""
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

Called once per saved step and appends one entry, which keeps the array aligned with
`sol.t`: `smoother_states[j]` holds the transition `sol.t[j] -> sol.t[j+1]`, so
`length(smoother_states) == length(sol.t) - 1`. The caller only runs this when
`OrdinaryDiffEqCore._savevalues!` actually saved; runs without a save (e.g. the final
step under `save_end=false`) belong to no stored transition and are skipped there.
"""
function _save_smoother_state!(smoother_states, cache)
    if iszero(cache.x_pred.Σ.R) || iszero(cache.measurement.Σ)
        # Degenerate step: `update!` skipped the update (zero predicted covariance,
        # mirroring `update!`'s own early-exit condition), or the measurement covariance
        # is exactly zero (e.g. `Σ_pred ≠ 0` but `H Σ_pred H' = 0` with no observation
        # noise). Either way there is no measurement information to smooth with, so the
        # backward pass at this step does a plain prediction (`ss === nothing` branch of
        # `_mbf_backward_step!`).
        push!(smoother_states, nothing)
        return nothing
    end
    T = eltype(cache.C_Dxd)
    H = copy(cache.H)
    z = Vector{T}(cache.measurement.μ)
    K = copy(cache.C_Dxd)
    S_U = _extract_measurement_chol(cache.C_dxd)
    # Snapshot the prior's rate parameter if it changes every step (IOUP with
    # `update_rate_parameter=true`): `calc_J!` mutates the array on the cache's prior at
    # the start of every step, so the backward pass needs a copy of this step's value.
    rate = if (cache.prior isa IOUP && cache.prior.update_rate_parameter)
        copy(cache.prior.rate_parameter)
    else
        nothing
    end
    push!(smoother_states, SmootherState(z, H, K, S_U, rate))
    return nothing
end

"""
    _extract_measurement_chol(C_dxd)

Extract the upper triangular Cholesky factor of the measurement covariance from `C_dxd`,
where `update!` computed it in-place via `cholesky!`. For 1×1 blocks (EK0, DiagonalEK1),
`update!` uses a scalar shortcut that skips the in-place Cholesky, so the factor is `√S`.

`C_dxd` holds the Cholesky factor in its upper triangle iff `update!` factorized it in
place; Dual eltypes (and any other eltype where `make_hermitian_if_fowarddiff` copies)
take the re-factorization branch below, since `update!` factorized a symmetrized copy
instead and left the raw (unfactorized) measurement covariance in `C_dxd`.
"""
_extract_measurement_chol(C_dxd::Matrix{T}) where {T} =
    length(C_dxd) == 1 ? fill(sqrt(C_dxd[1, 1]), 1, 1) :
    _extract_measurement_chol_dense(C_dxd, T)
_extract_measurement_chol_dense(C_dxd::Matrix, ::Type) = Matrix(UpperTriangular(C_dxd))
function _extract_measurement_chol_dense(
    C_dxd::Matrix, ::Type{<:ForwardDiff.Dual})
    F = cholesky(Symmetric(C_dxd), check=false)
    issuccess(F) || error(
        "Cholesky factorization of the measurement covariance failed; " *
        "cannot store the smoother state for backward smoothing.")
    return Matrix(F.U)
end
_extract_measurement_chol(C_dxd::IsometricKroneckerProduct) =
    IsometricKroneckerProduct(C_dxd.rdim, _extract_measurement_chol(C_dxd.B))
_extract_measurement_chol(C_dxd::BlocksOfDiagonals) =
    BlocksOfDiagonals([_extract_measurement_chol(b) for b in blocks(C_dxd)])

"""
    _measurement_chol_scale(mle_diffusion::Diagonal) -> Number or Diagonal

Pre-compute the measurement-Cholesky scale factor `M = sqrt.(mle_diffusion)` once, so
that the per-step [`_rescale_measurement_chol!`](@ref) loop avoids redundant allocations.
Returns a scalar for isotropic (FillArrays-backed) diffusions, a `Diagonal` otherwise.
"""
_measurement_chol_scale(D::Diagonal{<:Number,<:FillArrays.Fill}) = sqrt(D.diag.value)
_measurement_chol_scale(D::Diagonal) = Diagonal(sqrt.(D.diag))

"""
    _rescale_measurement_chol!(S_U, scale)

Rescale a stored smoother-state measurement-covariance Cholesky factor `S_U` after the
solution's covariances were calibrated (see [`calibrate_solution!`](@ref)). `scale` is
the pre-computed output of [`_measurement_chol_scale`](@ref): a scalar for isotropic
diffusions, a `Diagonal` for per-dimension ones (where the measurement Jacobians of
EK0 and DiagonalEK1 are dimension-pure). The calibrated factor is `S_U' = S_U * scale`.
"""
function _rescale_measurement_chol!(S_U::Matrix, σ::Number)
    rmul!(S_U, σ)
    return S_U
end
function _rescale_measurement_chol!(S_U::IsometricKroneckerProduct, σ::Number)
    rmul!(S_U.B, σ)
    return S_U
end
function _rescale_measurement_chol!(S_U::BlocksOfDiagonals, σ::Number)
    @simd ivdep for i in eachindex(blocks(S_U))
        rmul!(blocks(S_U)[i], σ)
    end
    return S_U
end
function _rescale_measurement_chol!(S_U::Matrix, M::Diagonal)
    rmul!(S_U, M)
    return S_U
end
function _rescale_measurement_chol!(S_U::BlocksOfDiagonals, M::Diagonal)
    @simd ivdep for i in eachindex(blocks(S_U))
        rmul!(blocks(S_U)[i], M.diag[i])
    end
    return S_U
end
function _rescale_measurement_chol!(S_U::IsometricKroneckerProduct, M::Diagonal)
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
    # `_savevalues!` returns `(saved, savedexactly)`; `saved` is `true` exactly when it
    # appended a new entry to `sol.t`. Without a save there is no solution index that the
    # current state belongs to, so the custom saves below have to be skipped: otherwise
    # they overwrite the last saved entry with the current (unsaved) state. This happens
    # e.g. with `save_end=false`, where the final step is taken but never saved.
    saved, _ = out

    # Save our custom stuff that we need for the posterior
    if saved && integ.opts.save_everystep
        i = integ.saveiter
        save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)
        _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x)
        copyat_or_push!(integ.sol.pu, i, integ.cache.pu_tmp)

        # Only stored when consumed; see `_needs_backward_kernels`.
        if _needs_backward_kernels(integ.alg)
            copyat_or_push!(
                integ.sol.backward_kernels, i, integ.cache.backward_kernel)
        end
        if integ.alg.smooth && integ.alg.smoother == :mbf
            # smoother_states[j] holds the transition sol.t[j] -> sol.t[j+1], so the step
            # just taken (ending at the freshly saved point i) is appended at i - 1.
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
