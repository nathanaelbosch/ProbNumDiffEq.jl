@doc raw"""
   invquad(v, M; v_cache, M_cache)

Compute ``v' M^{-1} v`` without allocations and with Matrix-specific specializations.

Needed for MLE diffusion estimation.
"""
invquad
invquad(v, M::Matrix; v_cache, M_cache) = begin
    v_cache .= v
    M = make_hermitian_if_fowarddiff(M)
    M_chol = cholesky!(copy!(M_cache, M))
    ldiv!(M_chol, v_cache)
    dot(v, v_cache)
end
invquad(v, M::IsometricKroneckerProduct; v_cache, M_cache=nothing) = begin
    v_cache .= v
    @assert length(M.B) == 1
    return dot(v, v_cache) / M.B[1]
end
invquad(v, M::BlocksOfDiagonals; v_cache, M_cache=nothing) = begin
    v_cache .= v
    @assert length(M.blocks) == length(v) == length(v_cache)
    @simd ivdep for i in eachindex(v)
        @assert length(M.blocks[i]) == 1
        @inbounds v_cache[i] /= M.blocks[i][1]
    end
    return dot(v, v_cache)
end

@doc raw"""
    estimate_global_diffusion(::FixedDiffusion, integ)

Updates the global quasi-MLE diffusion estimate on the current measuremnt.

The global quasi-MLE diffusion estimate Corresponds to
```math
\hat{σ}^2_N = \frac{1}{Nd} \sum_{i=1}^N z_i^T S_i^{-1} z_i,
```
where ``z_i, S_i`` are taken the predicted observations from each step.
This function updates the iteratively computed global diffusion estimate by computing
```math
\hat{σ}^2_n = \hat{σ}^2_{n-1} + ((z_n^T S_n^{-1} z_n) / d - \hat{σ}^2_{n-1}) / n.
```

For more background information
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
function estimate_global_diffusion(::FixedDiffusion, integ)
    @unpack d, measurement, m_tmp, Smat = integ.cache
    v, S = measurement.μ, measurement.Σ
    _v, _S = m_tmp.μ, m_tmp.Σ

    diffusion_increment = invquad(v, S; v_cache=_v, M_cache=_S) / d

    new_mle_diffusion = if integ.success_iter == 0
        diffusion_increment
    else
        current_mle_diffusion = integ.cache.global_diffusion.diag.value
        current_mle_diffusion +
        (diffusion_increment - current_mle_diffusion) / integ.success_iter
    end

    integ.cache.global_diffusion = new_mle_diffusion * Eye(d)
    return integ.cache.global_diffusion
end

@doc raw"""
    estimate_global_diffusion(::FixedMVDiffusion, integ)

Updates the multivariate global quasi-MLE diffusion estimate on the current measuremnt.

**This only works with the EK0!**

The global quasi-MLE diffusion estimate Corresponds to
```math
[\hat{Σ}^2_N]_{jj} = \frac{1}{N} \sum_{i=1}^N [z_i]_j^2 / [S_i]_{11},
```
where ``z_i, S_i`` are taken the predicted observations from each step.
This function updates the iteratively computed global diffusion estimate by computing
```math
[\hat{Σ}^2_n]_{jj} = [\hat{Σ}^2_{n-1}]_{jj} + ([z_n]_j^2 / [S_n]_{11}, - [\hat{Σ}^2_{n-1}]_{jj}) / n.
```

For more background information
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
function estimate_global_diffusion(::FixedMVDiffusion, integ)
    @unpack d, q, measurement, local_diffusion, C_d = integ.cache
    v, S = measurement.μ, measurement.Σ
    # @assert diag(S) |> unique |> length == 1
    diffusion_increment = let
        @.. C_d = v^2 / S[1, 1]
        Diagonal(C_d)
    end

    new_mle_diffusion = if integ.success_iter == 0
        diffusion_increment
    else
        current_mle_diffusion = integ.cache.global_diffusion
        @.. current_mle_diffusion +
            (diffusion_increment - current_mle_diffusion) / integ.success_iter
    end

    copy!(integ.cache.global_diffusion, new_mle_diffusion)
    return integ.cache.global_diffusion
end

@doc raw"""
    local_scalar_diffusion(integ)

Compute the local scalar quasi-MLE diffusion estimate.

Corresponds to
```math
σ² = zᵀ (H Q H^T)⁻¹ z,
```
where ``z, H, Q`` are taken from the passed integrator.

For more background information
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
function local_scalar_diffusion(cache)
    @unpack d, R, H, Qh, measurement, m_tmp, Smat, C_Dxd, C_d, C_dxd = cache
    z = measurement.μ
    HQH = let
        _matmul!(C_Dxd, Qh.R, H')
        _matmul!(C_dxd, C_Dxd', C_Dxd)
    end
    σ² = invquad(z, HQH; v_cache=C_d, M_cache=C_dxd) / d
    cache.local_diffusion = σ² * Eye(d)
    return cache.local_diffusion
end

@doc raw"""
    local_diagonal_diffusion(cache)

Compute the local diagonal quasi-MLE diffusion estimate.

**This only works with the EK0!**

Corresponds to
```math
Σ_{ii} = z_i^2 / (H Q H^T)_{ii},
```
where ``z, H, Q`` are taken from the passed integrator.

For more background information
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
function local_diagonal_diffusion(cache)
    @unpack d, q, H, Qh, measurement, m_tmp = cache
    tmp = m_tmp.μ
    @unpack local_diffusion = cache
    @assert (H == cache.E1) || (H == cache.E2) || H isa BlocksOfDiagonals

    z = measurement.μ
    # HQH = H * unfactorize(Qh) * H'
    # @assert HQH |> diag |> unique |> length == 1
    # c1 = view(_matmul!(cache.C_Dxd, Qh.R, H'), :, 1)
    # Q_11 = dot(c1, c1)

    Q_11 = if Qh.R isa BlocksOfDiagonals
        for i in 1:d
            c1 = _matmul!(
                view(cache.C_Dxd.blocks[i], :, 1:1),
                Qh.R.blocks[i],
                view(H.blocks[i], 1:1, :)',
            )
            tmp[i] = dot(c1, c1)
        end
        tmp
    else
        @warn "This is not yet implemented efficiently; TODO"
        diag(X_A_Xt(Qh, H))
    end

    @. local_diffusion.diag = z^2 / Q_11
    return local_diffusion
end
