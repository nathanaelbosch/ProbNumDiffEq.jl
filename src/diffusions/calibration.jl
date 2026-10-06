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
invquad(v, M::PSDMatrix; v_cache, M_cache) =
    invquad(v, _matmul!(M_cache, M.R', M.R); v_cache, M_cache)

@doc raw"""
    estimate_global_diffusion(::FixedDiffusion, integ, z, S, ztSinvz)

Updates the global quasi-MLE diffusion estimate on the current measuremnt.

The global quasi-MLE diffusion estimate Corresponds to
```math
\hat{σ}^2_N = \frac{1}{Nd} \sum_{i=1}^N z_i^T S_i^{-1} z_i,
```
where ``z_i`` is the residual of the observation in each step and ``S_i`` its covariance;
[`update!`](@ref) returns ``z_i^T S_i^{-1} z_i`` as `ztSinvz`.
This function updates the iteratively computed global diffusion estimate by computing
```math
\hat{σ}^2_n = \hat{σ}^2_{n-1} + ((z_n^T S_n^{-1} z_n) / d - \hat{σ}^2_{n-1}) / n.
```

For more background information
* [Bosch et al. (2021)](@cite bosch20capos) "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS
"""
function estimate_global_diffusion(::FixedDiffusion, integ, z, S, ztSinvz)
    diffusion_increment = ztSinvz / integ.cache.d

    new_mle_diffusion = if integ.success_iter == 0
        diffusion_increment
    else
        current_mle_diffusion = integ.cache.global_diffusion
        current_mle_diffusion +
        (diffusion_increment - current_mle_diffusion) / integ.success_iter
    end

    integ.cache.global_diffusion = new_mle_diffusion
    return integ.cache.global_diffusion
end

@doc raw"""
    estimate_global_diffusion(::FixedMVDiffusion, integ, z, S, ztSinvz)

Updates the multivariate global quasi-MLE diffusion estimate on the current measuremnt.

Used for `FixedMVDiffusion(calibrate=true)`, which requires block-diagonal covariances.

The global quasi-MLE diffusion estimate Corresponds to
```math
[\hat{Σ}^2_N]_{jj} = \frac{1}{N} \sum_{i=1}^N [z_i]_j^2 / [S_i]_{jj},
```
where ``z_i`` is the residual of the observation in each step and ``S_i`` its covariance
computed by [`update!`](@ref).
This function updates the iteratively computed global diffusion estimate by computing
```math
[\hat{Σ}^2_n]_{jj} = [\hat{Σ}^2_{n-1}]_{jj} + ([z_n]_j^2 / [S_n]_{jj} - [\hat{Σ}^2_{n-1}]_{jj}) / n.
```

For more background information
* [Bosch et al. (2021)](@cite bosch20capos) "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS
"""
function estimate_global_diffusion(::FixedMVDiffusion, integ, z, S, ztSinvz)
    @unpack C_d = integ.cache
    diffusion_increment = let
        diag!(C_d, S)
        @.. C_d = z^2 / C_d
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
    local_scalar_diffusion(cache, z, HQH)

Compute and return the local scalar quasi-MLE diffusion estimate as a `Number`.

Corresponds to
```math
σ² = zᵀ (H Q H^T)⁻¹ z / d,
```
where ``z`` is the residual of the [`LinearizedObservation`](@ref) and `HQH` is computed by
[`observed_process_noise!`](@ref).

For more background information
* [Bosch et al. (2021)](@cite bosch20capos) "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS
"""
function local_scalar_diffusion(cache, z, HQH)
    @unpack d, C_d, C_dxd = cache
    σ² = invquad(z, HQH; v_cache=C_d, M_cache=C_dxd) / d
    return σ²
end

@doc raw"""
    local_diagonal_diffusion(cache, z, HQH)

Compute the local diagonal quasi-MLE diffusion estimate.

Only valid for block-diagonal covariances, where `H` does not couple the dimensions.

Corresponds to
```math
Σ_{ii} = z_i^2 / (H Q H^T)_{ii},
```
where ``z`` is the residual of the [`LinearizedObservation`](@ref) and `HQH` is computed by
[`observed_process_noise!`](@ref).

For more background information
* [Bosch et al. (2021)](@cite bosch20capos) "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS
"""
function local_diagonal_diffusion(cache, z, HQH)
    @unpack local_diffusion = cache
    HQH_diag = diag!(local_diffusion.diag, HQH)
    @.. local_diffusion.diag = z^2 / HQH_diag
    return local_diffusion
end
