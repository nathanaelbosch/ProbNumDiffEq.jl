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

Uses the square-root Modified Bryson-Frazier (√MBF) smoother (Gibbs 2011), which propagates
adjoint variables backward and only inverts the `d×d` measurement covariance (not the full
`D×D` predicted state covariance as in the RTS smoother). The λ/Λ update and predict recursion
is done in sqrt form in each step's local preconditioned coordinates for numerical stability.
Covariance recovery uses a hyperbolic QR factorization to guarantee a positive semi-definite
result, also computed in preconditioned coordinates.

The actual per-step work is done by [`_mbf_backward_step!`](@ref), which dispatches on the
structure of the state covariances (dense, EK0's Kronecker, or DiagonalEK1's block-diagonal),
the same way [`predict_cov!`](@ref) and [`update!`](@ref) do -- so this loop doesn't need to
know or care which algorithm produced the solution being smoothed.
"""
function smooth_solution!(integ)
    @unpack cache, sol = integ
    for (i, x) in enumerate(sol.x_filt)
        copyat_or_push!(sol.x_smooth, i, x)
    end

    @unpack x_smooth, t, diffusions, smoother_states = sol
    @unpack x_pred, C_DxD, C_2DxD = cache
    n = length(x_smooth)

    λ = zeros(eltype(x_smooth[1].μ), length(x_smooth[1].μ))
    # U_Λ needs to hold a general (non-diagonal) matrix once measurement info accumulates,
    # so it must be zero-initialized from the state covariance's structure (dense/Kronecker/
    # block-diagonal), not from `cache.P`/`PI` -- those are diagonal preconditioners even in
    # the dense (EK1) case, and a `Diagonal`-typed `U_Λ` can't hold the general update result.
    U_Λ = zero(x_smooth[1].Σ.R)

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

        ss_idx = i - 1
        dt_step = t[i] - t[i-1]
        make_transition_matrices!(cache, cache.prior, dt_step)

        ss =
            (ss_idx < 1 || ss_idx > length(smoother_states)) ? nothing :
            smoother_states[ss_idx]
        if !isnothing(ss)
            # Recompute Σ_pred (and hence K, S_U, inside _mbf_backward_step!) fresh from
            # the (possibly diffusion-recalibrated) filtered covariance, so it is always
            # consistent with it -- unlike z, H (which do not depend on the diffusion
            # scale), K and S_U cannot simply be reused from the forward pass if the
            # diffusion was later calibrated.
            predict_mean!(x_pred.μ, sol.x_filt[i-1].μ, cache.Ah)
            extrapolation_diff = diffusions[min(i - 1, length(diffusions))]
            predict_cov!(
                x_pred.Σ, sol.x_filt[i-1].Σ, cache.Ah, cache.Qh, C_DxD, C_2DxD,
                extrapolation_diff)
        end

        λ, U_Λ = _mbf_backward_step!(
            x_smooth[i-1], λ, U_Λ, sol.x_filt[i-1], ss, x_pred,
            cache.P, cache.PI, cache.A, cache.R)

        _gaussian_mul!(sol.pu[i-1], cache.SolProj, x_smooth[i-1])
        sol.u[i-1][:] .= sol.pu[i-1].μ
    end
    return nothing
end

"""
    _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, x_pred, P, PI, A, R)

One backward step of the √MBF recursion: incorporate the measurement info in `ss` (or just
predict, if `ss === nothing`) into the adjoint state `(λ, U_Λ)`, then recover the smoothed
mean/covariance into `x_smooth_prev` using the filtered state `x_filt_prev`. Returns the
updated `(λ, U_Λ)` for use at the next (earlier) step.

Dispatches on the structure of `P` (which matches that of the state covariances) so that EK0's
and DiagonalEK1's efficient Kronecker/block-diagonal representations are preserved: the λ/Λ
recursion and covariance recovery only ever touch the small `(q+1)×(q+1)` blocks (batched over
the `d` ODE dimensions for EK0, or looped independently for DiagonalEK1), never a dense `D×D`
matrix -- without this, backward smoothing would cost `O(D^3) = O((d(q+1))^3)` instead of the
`O(d)`-ish cost the rest of the package achieves for these algorithms.

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
function _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, x_pred, P, PI, A, R)
    D = length(λ)
    if isnothing(ss)
        λ_new = A' * λ
        U_Λ_new = U_Λ * A
    else
        K, S_U = _mbf_measurement_covariance(ss.H, x_pred.Σ, R)

        H_prec = ss.H * PI'
        K_prec = P * K
        λ_prec = PI' * λ
        U_Λ_prec = U_Λ * PI

        _sqrt_mbf_update!(λ_prec, U_Λ_prec, K_prec, S_U, ss.z, H_prec, D)

        λ_prec = A' * λ_prec
        U_Λ_prec = U_Λ_prec * A

        λ_new = P' * λ_prec
        U_Λ_new = U_Λ_prec * P
    end

    # Recover smoothed mean/covariance using the now-updated λ, Λ (information from
    # measurements {i, ..., n}).
    U_f = Matrix(x_filt_prev.Σ.R)
    _mbf_recover_mean!(x_smooth_prev.μ, U_f, λ_new)
    U_f_prec = U_f * P'
    U_Λ_prec_rec = U_Λ_new * PI
    new_R_prec = _mbf_recover_cov(U_f_prec, U_Λ_prec_rec)
    new_R = new_R_prec * PI'
    # `new_R` is only upper triangular up to floating-point roundoff, which -- while tiny
    # relative to the overall matrix -- can be large relative to the (possibly much
    # smaller, measurement-shrunk) smoothed covariance entries. Naively dropping the
    # lower-triangular part (as a plain truncation would) therefore changes `new_R'*new_R`
    # non-negligibly; re-triangularize via QR instead, which preserves `R'R` exactly.
    new_R = _positive_qr_r(new_R)
    _copy_upper_to_R!(x_smooth_prev.Σ.R, new_R)

    return λ_new, U_Λ_new
end

# Kronecker version (EK0): reduce to the shared small `.B` block, batched over the `d` ODE
# dimensions via `(q+1)×d`-shaped λ, instead of densifying to `D×D`.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ::IsometricKroneckerProduct, x_filt_prev, ss, x_pred,
    P::IsometricKroneckerProduct, PI::IsometricKroneckerProduct,
    A::IsometricKroneckerProduct, R,
)
    d = P.rdim
    Q = size(P.B, 1)
    _λ = reshape_no_alloc(λ, d, Q)'
    P_B, PI_B, A_B, U_Λ_B = P.B, PI.B, A.B, U_Λ.B

    if isnothing(ss)
        λ_new_B = A_B' * _λ
        U_Λ_new_B = U_Λ_B * A_B
    else
        K_B, S_U_B = _mbf_measurement_covariance(ss.H, x_pred.Σ, R)

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
    U_f_prec = U_f_B * P_B'
    U_Λ_prec_rec = U_Λ_new_B * PI_B
    new_R_B = _mbf_recover_cov(U_f_prec, U_Λ_prec_rec) * PI_B'
    new_R_B = _positive_qr_r(new_R_B)
    copy!(x_smooth_prev.Σ.R.B, UpperTriangular(new_R_B))

    λ_new = similar(λ)
    reshape_no_alloc(λ_new, d, Q)' .= λ_new_B
    return λ_new, IsometricKroneckerProduct(d, U_Λ_new_B)
end

# Block-diagonal version (DiagonalEK1): unlike EK0, each of the `d` ODE dimensions has its own
# measurement Jacobian (and hence its own Kalman gain), so there is no shared small block to
# batch over. Instead, reduce to `d` independent `(q+1)`-sized problems, one per diagonal
# block, and recurse -- each recursive call dispatches to the dense method above.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ::BlocksOfDiagonals, x_filt_prev, ss, x_pred,
    P::BlocksOfDiagonals, PI::BlocksOfDiagonals, A::BlocksOfDiagonals, R,
)
    d = length(blocks(P))
    D = length(λ)
    λ_new = similar(λ)
    new_U_Λ_blocks = similar(blocks(U_Λ))

    for bi in 1:d
        _ss =
            isnothing(ss) ? nothing :
            SmootherState(collect(view(ss.z, bi:bi)), ss.H.blocks[bi])
        _x_filt_prev =
            Gaussian(view(x_filt_prev.μ, bi:d:D), PSDMatrix(x_filt_prev.Σ.R.blocks[bi]))
        _x_smooth_prev =
            Gaussian(view(x_smooth_prev.μ, bi:d:D), PSDMatrix(x_smooth_prev.Σ.R.blocks[bi]))
        _x_pred =
            isnothing(ss) ? nothing :
            Gaussian(view(x_pred.μ, bi:d:D), PSDMatrix(x_pred.Σ.R.blocks[bi]))
        _R = isnothing(R) ? nothing : PSDMatrix(R.R.blocks[bi])

        λ_bi_new, U_Λ_bi_new = _mbf_backward_step!(
            _x_smooth_prev, collect(view(λ, bi:d:D)), U_Λ.blocks[bi],
            _x_filt_prev, _ss, _x_pred, P.blocks[bi], PI.blocks[bi], A.blocks[bi], _R)

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
covariance is much smaller than the filtered one). Must be called with `U_f`, `U_Λ`
in a locally well-scaled (preconditioned) coordinate system -- see the call site in
`smooth_solution!`.
"""
function _mbf_recover_cov(U_f::Matrix, U_Λ::Matrix)
    V = U_Λ * U_f'
    R_M = _hyperbolic_qr!(V)
    return R_M * U_f
end

function _hyperbolic_qr!(V::Matrix{T}) where {T}
    d = size(V, 1)
    top = Matrix{T}(I, d, d)
    bot = V
    v_h = Vector{T}(undef, d)
    w_h = Vector{T}(undef, d)

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

function _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D)
    S_chol = Cholesky(S_U, 'U', 0)

    # λ update (BK-free form); z_code = h(m_pred), innovation = -z_code
    w = K' * λ - S_chol \ z
    λ .-= H' * w

    # Λ update (sqrt form via QR).
    # BK = I - K*H; the second stack block must be a factor M with M'M = S⁻¹. With the upper
    # triangular Cholesky factor `S_U` (S = S_U'S_U) that is `S_U' \ H = S_U⁻ᵀH`, since
    # (S_U⁻ᵀH)'(S_U⁻ᵀH) = H'S⁻¹H; using `S_U \ H = S_U⁻¹H` instead would give
    # H'(S_U S_U')⁻¹H ≠ H'S⁻¹H (unless S_U is diagonal) and silently corrupt Λ.
    BK = -K * H
    @inbounds for j in 1:D
        BK[j, j] += 1
    end
    Stack = [U_Λ * BK; S_chol.U' \ H]
    U_Λ .= _positive_qr_r(Stack)
end

function _save_smoother_state!(smoother_states, cache)
    T = eltype(cache.C_Dxd)
    H = copy(cache.H)
    z = Vector{T}(cache.measurement.μ)
    push!(smoother_states, SmootherState(z, H))
end

"""
    _mbf_measurement_covariance(H, Σ_pred, R)

Compute the Kalman gain `K` and measurement-covariance Cholesky factor `S_U` (upper
triangular) from `H` and the predicted covariance `Σ_pred`, freshly at backward-smoothing
time so that they are always consistent with (possibly diffusion-recalibrated) covariances.
"""
function _mbf_measurement_covariance(H, Σ_pred::PSDMatrix, R)
    T = eltype(H)
    P_pred = Matrix(Σ_pred)
    S_mat = H * P_pred * H'
    if !isnothing(R)
        S_mat .+= R.R' * R.R
    end
    S_chol = cholesky(Symmetric(S_mat), check=false)
    S_U = issuccess(S_chol) ? Matrix{T}(S_chol.U) : Matrix{T}(I, size(S_mat)...)
    K =
        issuccess(S_chol) ? (S_chol \ (P_pred * H')')' :
        zeros(T, size(P_pred, 1), size(H, 1))
    return K, S_U
end

# Kronecker version: reduce to the shared small `.B` block instead of densifying to `D×D`.
function _mbf_measurement_covariance(
    H::IsometricKroneckerProduct, Σ_pred::PSDMatrix{T,<:IsometricKroneckerProduct}, R,
) where {T}
    _R = isnothing(R) ? nothing : PSDMatrix(R.R.B)
    return _mbf_measurement_covariance(H.B, PSDMatrix(Σ_pred.R.B), _R)
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

function _copy_upper_to_R!(R::Matrix, U_s)
    n = size(U_s, 1)
    copy!(R, UpperTriangular(view(U_s, 1:n, 1:n)))
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

        if integ.alg.smooth
            copyat_or_push!(
                integ.sol.backward_kernels, i, integ.cache.backward_kernel)
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
