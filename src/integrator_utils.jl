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
`D×D` predicted state covariance as in the RTS smoother). Covariance recovery uses hyperbolic
QR to guarantee positive semi-definiteness.
"""
function smooth_solution!(integ)
    @unpack cache, sol = integ
    for (i, x) in enumerate(sol.x_filt)
        copyat_or_push!(sol.x_smooth, i, x)
    end

    @unpack x_smooth, t, diffusions, smoother_states = sol
    @unpack d, q, x_pred, C_DxD, C_2DxD = cache
    D = d * (q + 1)
    n = length(x_smooth)

    λ = zeros(eltype(x_smooth[1].μ), D)
    λ_tmp = similar(λ)
    U_Λ = zeros(eltype(λ), D, D)
    U_Λ_tmp = similar(U_Λ)

    for i in n:-1:1
        if i < n && iszero(t[i+1] - t[i])
            copy!(x_smooth[i], x_smooth[i+1])
            _gaussian_mul!(sol.pu[i], cache.SolProj, x_smooth[i])
            sol.u[i][:] .= sol.pu[i].μ
            continue
        end

        # Recover smoothed mean: m_s = m_f - P_f * λ
        U_f = Matrix(sol.x_filt[i].Σ.R)
        _mbf_recover_mean!(x_smooth[i].μ, U_f, λ)

        # Recover smoothed covariance via hyperbolic QR
        if i < n
            _mbf_recover_cov!(x_smooth[i].Σ, U_f, U_Λ)
        end

        _gaussian_mul!(sol.pu[i], cache.SolProj, x_smooth[i])
        sol.u[i][:] .= sol.pu[i].μ

        if i == 1
            break
        end

        ss_idx = i - 1
        dt_step = t[i] - t[i-1]
        make_transition_matrices!(cache, cache.prior, dt_step)

        if ss_idx < 1 || ss_idx > length(smoother_states)
            copyto!(λ_tmp, λ)
            _matmul!(λ, cache.Ah', λ_tmp)
            U_Λ .= U_Λ * Matrix(cache.Ah)
            continue
        end
        ss = smoother_states[ss_idx]

        # Recompute Σ_pred (and hence K, S_U) fresh from the (possibly diffusion-
        # recalibrated) filtered covariance, so it is always consistent with it --
        # unlike z, H (which do not depend on the diffusion scale), K and S_U cannot
        # simply be reused from the forward pass if the diffusion was later calibrated.
        predict_mean!(x_pred.μ, sol.x_filt[i-1].μ, cache.Ah)
        extrapolation_diff = diffusions[min(i - 1, length(diffusions))]
        predict_cov!(
            x_pred.Σ, sol.x_filt[i-1].Σ, cache.Ah, cache.Qh, C_DxD, C_2DxD,
            extrapolation_diff)
        K, S_U = _mbf_measurement_covariance(ss.H, x_pred.Σ, cache.R)

        # √MBF adjoint update
        _sqrt_mbf_update!(λ, U_Λ, K, S_U, ss.z, ss.H, D, d)

        # √MBF predict: propagate backward through transition
        Ah_mat = Matrix(cache.Ah)
        copyto!(λ_tmp, λ)
        λ .= Ah_mat' * λ_tmp
        copyto!(U_Λ_tmp, U_Λ)
        U_Λ .= U_Λ_tmp * Ah_mat
    end
    return nothing
end

function _mbf_recover_mean!(μ_smooth, U_f::Matrix, λ)
    μ_smooth .-= U_f' * (U_f * λ)
end

function _mbf_recover_cov!(Σ_smooth::PSDMatrix, U_f::Matrix, U_Λ::Matrix)
    # P_s = P_f - P_f*Λ*P_f = U_f' * (I - W*W') * U_f  with  W = U_f*U_Λ'
    # _hyperbolic_qr! computes R with R'R = I - V'V, so pass V = W' = U_Λ*U_f'
    V = U_Λ * U_f'
    R_M = _hyperbolic_qr!(V)
    U_s = R_M * U_f
    _copy_upper_to_R!(Σ_smooth.R, U_s)
end

function _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D, d)
    S_chol = Cholesky(S_U, 'U', 0)

    # λ update (BK-free form); z_code = h(m_pred), innovation = -z_code
    w = K' * λ - S_chol \ z
    λ .-= H' * w

    # Λ update (sqrt form via QR)
    BK = -K * H
    @inbounds for j in 1:D
        BK[j, j] += 1
    end
    Stack = [U_Λ * BK; S_chol.U \ H]
    U_Λ .= _positive_qr_r(Stack)
end

function _save_smoother_state!(smoother_states, cache)
    T = eltype(cache.C_Dxd)
    H = Matrix{T}(cache.H)
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
    K = issuccess(S_chol) ? (S_chol \ (P_pred * H')')' : zeros(T, size(P_pred, 1), size(H, 1))
    return K, S_U
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
        for l in i:d; asq += top[l, i] * top[l, i]; end
        bsq = zero(T)
        for l in 1:d; bsq += bot[l, i] * bot[l, i]; end

        α² = max(asq - bsq, zero(T))
        α = sqrt(α²)

        if α < eps(T) * d
            for l in i:d; top[l, i] = zero(T); end
            for l in 1:d; bot[l, i] = zero(T); end
            continue
        end

        α = top[i, i] >= 0 ? -α : α
        β = one(T) / (α * (α - top[i, i]))

        # u_top, u_bot (unscaled householder-like vectors for this hyperbolic reflection)
        v_h[1] = top[i, i] - α
        for l in 2:na; v_h[l] = top[i + l - 1, i]; end
        for l in 1:d; w_h[l] = bot[l, i]; end

        for j in (i + 1):d
            τ = zero(T)
            for l in 1:na; τ += v_h[l] * top[i + l - 1, j]; end
            for l in 1:d; τ -= w_h[l] * bot[l, j]; end
            τ *= β

            for l in 1:na; top[i + l - 1, j] -= τ * v_h[l]; end
            for l in 1:d; bot[l, j] -= τ * w_h[l]; end
        end

        top[i, i] = α
        for l in (i + 1):d; top[l, i] = zero(T); end
        for l in 1:d; bot[l, i] = zero(T); end
    end
    return top
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

function _copy_upper_to_R!(R, U_s)
    if R isa IsometricKroneckerProduct
        d = R.rdim
        q1 = size(R.B, 1)
        for j in 1:q1, i in 1:q1
            R.B[i, j] = U_s[(i-1)*d+1, (j-1)*d+1]
        end
    elseif R isa BlocksOfDiagonals
        d = length(blocks(R))
        q1 = size(blocks(R)[1], 1)
        for bi in 1:d
            for j in 1:q1, i in 1:q1
                blocks(R)[bi][i, j] = U_s[(i-1)*d+bi, (j-1)*d+bi]
            end
        end
    else
        n = size(U_s, 1)
        copy!(R, UpperTriangular(view(U_s, 1:n, 1:n)))
    end
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
