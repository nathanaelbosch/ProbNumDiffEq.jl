abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true

apply_diffusion(Q::PSDMatrix, diffusion::Diagonal) = X_A_Xt(Q, sqrt.(diffusion))
apply_diffusion(Q::PSDMatrix, diffusion::Number) = PSDMatrix(Q.R * sqrt.(diffusion))

estimate_global_diffusion(diffusion::AbstractDynamicDiffusion, d, q, Eltype) = NaN

"""
    DynamicDiffusion()

Time-varying, isotropic diffusion, which is quasi-maximum-likelihood-estimated at each step.

**This is the recommended diffusion when using adaptive step-size selection,** and in
particular also when solving stiff systems.
"""
struct DynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicDiffusion, d, q, Eltype) = one(Eltype)
estimate_local_diffusion(::DynamicDiffusion, integ) = local_scalar_diffusion(integ.cache)

"""
    DynamicMVDiffusion()

Time-varying, diagonal diffusion, which is quasi-maximum-likelihood-estimated at each step.

**Only works with the [`EK0`](@ref)!**

A multi-variate version of [`DynamicDiffusion`](@ref), where instead of an isotropic matrix,
a diagonal matrix is estimated. This can be helpful to get more expressive posterior
covariances when using the [`EK0`](@ref), since the individual dimensions can be adjusted
separately.

# References
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
struct DynamicMVDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicMVDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
estimate_local_diffusion(::DynamicMVDiffusion, integ) =
    local_diagonal_diffusion(integ.cache)

"""
    FixedDiffusion(; initial_diffusion=1.0, calibrate=true)

Time-fixed, isotropic diffusion, which is (optionally) quasi-maximum-likelihood-estimated.

**This is the recommended diffusion when using fixed steps.**

By default with `calibrate=true`, all covariances are re-scaled at the end of the solve
with the MLE diffusion. Set `calibrate=false` to skip this step, e.g. when setting the
`initial_diffusion` and then estimating the diffusion outside of the solver
(e.g. with [Fenrir.jl](https://github.com/nathanaelbosch/Fenrir.jl)).
"""
Base.@kwdef struct FixedDiffusion{T<:Number} <: AbstractStaticDiffusion
    initial_diffusion::T = 1.0
    calibrate::Bool = true
end
initial_diffusion(diffusionmodel::FixedDiffusion, d, q, Eltype) =
    diffusionmodel.initial_diffusion * one(Eltype)
estimate_local_diffusion(::FixedDiffusion, integ) = local_scalar_diffusion(integ.cache)
function estimate_global_diffusion(::FixedDiffusion, integ)
    @unpack d, measurement, m_tmp, Smat = integ.cache
    # sol_diffusions = integ.sol.diffusions

    v, S = measurement.μ, measurement.Σ
    e = m_tmp.μ
    _S = S
    e .= v
    diffusion_t = if _S isa IsometricKroneckerProduct
        @assert length(_S.B) == 1
        dot(v, e) / d / _S.B[1]
    else
        S_chol = cholesky!(_S)
        ldiv!(S_chol, e)
        dot(v, e) / d
    end

    if integ.success_iter == 0
        # @assert length(sol_diffusions) == 0
        global_diffusion = diffusion_t
        return global_diffusion
    else
        # @assert length(sol_diffusions) == integ.success_iter
        diffusion_prev = integ.cache.global_diffusion
        global_diffusion =
            diffusion_prev + (diffusion_t - diffusion_prev) / integ.success_iter
        # @info "compute diffusion" diffusion_prev global_diffusion
        return global_diffusion
    end
end

"""
    FixedMVDiffusion(; initial_diffusion=1.0, calibrate=true)

Time-fixed, diagonal diffusion, which is quasi-maximum-likelihood-estimated at each step.

**Only works with the [`EK0`](@ref)!**

A multi-variate version of [`FixedDiffusion`](@ref), where instead of an isotropic matrix,
a diagonal matrix is estimated. This can be helpful to get more expressive posterior
covariances when using the [`EK0`](@ref), since the individual dimensions can be adjusted
separately.

# References
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
Base.@kwdef struct FixedMVDiffusion{T} <: AbstractStaticDiffusion
    initial_diffusion::T = 1.0
    calibrate::Bool = true
end
function initial_diffusion(diffusionmodel::FixedMVDiffusion, d, q, Eltype)
    initdiff = diffusionmodel.initial_diffusion
    @assert initdiff isa Number || length(initdiff) == d
    return kron(Diagonal(initdiff .* ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
end
estimate_local_diffusion(::FixedMVDiffusion, integ) = local_diagonal_diffusion(integ.cache)
function estimate_global_diffusion(::FixedMVDiffusion, integ)
    @unpack d, q, measurement, local_diffusion = integ.cache

    v, S = measurement.μ, measurement.Σ
    # S_11 = diag(S)[1]
    S_11 = S[1, 1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(Σ, I(q + 1))

    if integ.success_iter == 0
        # @assert length(diffusions) == 0
        return Σ_out
    else
        # @assert length(diffusions) == integ.success_iter
        diffusion_prev = integ.cache.global_diffusion
        diffusion =
            @. diffusion_prev =
                diffusion_prev + (Σ_out - diffusion_prev) / integ.success_iter
        return diffusion
    end
end

"""
    local_scalar_diffusion(integ)

Compute the local, scalar diffusion estimate.

Corresponds to
```math
σ² = zᵀ (H Q H^T)⁻¹ z,
```
where ``z, H, Q`` are taken from the passed integrator.

For more background information
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
function local_scalar_diffusion(cache)
    @unpack d, R, H, Qh, measurement, m_tmp, Smat, C_Dxd = cache
    z = measurement.μ
    e, HQH = m_tmp.μ, m_tmp.Σ
    _matmul!(C_Dxd, Qh.R, H')
    HQHmat = _matmul!(Smat, C_Dxd', C_Dxd)
    e .= z
    σ² = if HQHmat isa IsometricKroneckerProduct
        @assert length(HQHmat.B) == 1
        dot(z, e) / d / HQHmat.B[1]
    else
        C = cholesky!(HQHmat)
        ldiv!(C, e)
        dot(z, e) / d
    end
    return σ²
end

"""
    local_diagonal_diffusion(cache)

Compute the local, scalar diffusion estimate.

Corresponds to
```math
Σ_{ii} = z_i^2 / (H Q H^T)_{ii},
```
where ``z, H, Q`` are taken from the passed integrator.
**This should only be used with the EK0!**

For more background information
* [bosch20capos](@cite) Bosch et al, "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS (2021)
"""
function local_diagonal_diffusion(cache)
    @unpack d, q, H, Qh, measurement, m_tmp, tmp = cache
    @unpack local_diffusion = cache
    z = measurement.μ
    HQHR = _matmul!(cache.C_Dxd, Qh.R, H')
    # Q0_11 = diag(HQH)[1]
    c1 = view(HQHR, :, 1)
    Q0_11 = dot(c1, c1)

    Σ_ii = @. m_tmp.μ = z^2 / Q0_11
    # Σ_ii .= max.(Σ_ii, eps(eltype(Σ_ii)))
    Σ = Diagonal(Σ_ii)

    # local_diffusion = kron(Σ, I(q+1))
    for i in 1:d
        for j in (i-1)*(q+1)+1:i*(q+1)
            local_diffusion[j, j] = Σ[i, i]
        end
    end
    return local_diffusion
end
