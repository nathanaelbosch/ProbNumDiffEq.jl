abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true

estimate_global_diffusion(diffusion::AbstractDynamicDiffusion, d, q, Eltype) = NaN

"""
    DynamicDiffusion()

**Recommended with adaptive steps**

A local diffusion parameter is estimated at each step. Works well with adaptive steps.
"""
struct DynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicDiffusion, d, q, Eltype) = one(Eltype)
estimate_local_diffusion(::DynamicDiffusion, integ) = local_scalar_diffusion(integ.cache)

"""
    DynamicMVDiffusion()

**Only works with the [`EK0`](@ref)**

A multi-variate version of [`DynamicDiffusion`](@ref), where instead of a scalar a
vector-valued diffusion is estimated. When using the EK0, this can be helpful when the
scales of the different dimensions vary a lot.
"""
struct DynamicMVDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicMVDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
estimate_local_diffusion(::DynamicMVDiffusion, integ) =
    local_diagonal_diffusion(integ.cache)

"""
    FixedDiffusion(; initial_diffusion=1.0, calibrate=true)

**Recommended with fixed steps**

If `calibrate=true`, the probabilistic solution is calibrated once at the end of the solve.
An initial diffusion parameter can be passed, but this only has an effect if
`calibrate=false` - which is typically not recommended.
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
    _S = _matmul!(Smat, S.R', S.R)
    S_chol = cholesky!(_S)
    e .= v
    ldiv!(S_chol, e)
    diffusion_t = dot(v, e) / d

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

**Only works with the [`EK0`](@ref)**

A multi-variate version of [`FixedDiffusion`](@ref), where instead of a scalar a
vector-valued diffusion is estimated. When using the EK0, this can be helpful when the
scales of the different dimensions vary a lot.
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
    c1 = view(S.R, :, 1)
    S_11 = dot(c1, c1)

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
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
"""
function local_scalar_diffusion(cache)
    @unpack d, R, H, Qh, measurement, m_tmp, Smat = cache
    z = measurement.μ
    e, HQH = m_tmp.μ, m_tmp.Σ
    fast_X_A_Xt!(HQH, Qh, H)
    HQHmat = _matmul!(Smat, HQH.R', HQH.R)
    C = cholesky!(HQHmat)
    e .= z
    ldiv!(C, e)
    σ² = dot(z, e) / d
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
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
"""
function local_diagonal_diffusion(cache)
    @unpack d, q, H, Qh, measurement, m_tmp, tmp = cache
    @unpack local_diffusion = cache
    z = measurement.μ
    HQH = fast_X_A_Xt!(m_tmp.Σ, Qh, H)
    # Q0_11 = diag(HQH)[1]
    c1 = view(HQH.R, :, 1)
    Q0_11 = dot(c1, c1)

    Σ_ii = @. tmp = z^2 / Q0_11
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
