abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true

estimate_global_diffusion(diffusion::AbstractDynamicDiffusion, d, q, Eltype) = NaN

struct DynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(diffusion::DynamicDiffusion, d, q, Eltype) = one(Eltype)
estimate_local_diffusion(kind::DynamicDiffusion, integ) = local_scalar_diffusion(integ)

struct DynamicMVDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(diffusionmodel::DynamicMVDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
estimate_local_diffusion(kind::DynamicMVDiffusion, integ) = local_diagonal_diffusion(integ)

Base.@kwdef struct FixedDiffusion{T<:Number} <: AbstractStaticDiffusion
    initial_diffusion::T = 1.0
    calibrate::Bool = true
end
initial_diffusion(diffusionmodel::FixedDiffusion, d, q, Eltype) =
    diffusionmodel.initial_diffusion * one(Eltype)
estimate_local_diffusion(kind::FixedDiffusion, integ) = local_scalar_diffusion(integ)
function estimate_global_diffusion(rule::FixedDiffusion, integ)
    @unpack d, measurement, m_tmp = integ.cache
    # sol_diffusions = integ.sol.diffusions

    v, S = measurement.μ, measurement.Σ
    e, _ = m_tmp.μ, m_tmp.Σ.mat
    _S = copy!(m_tmp.Σ.mat, S.mat)
    if _S isa Diagonal
        e .= v ./ _S.diag
    else
        S_chol = cholesky!(_S)
        ldiv!(e, S_chol, v)
    end
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

Base.@kwdef struct FixedMVDiffusion{T} <: AbstractStaticDiffusion
    initial_diffusion::T = 1.0
    calibrate::Bool = true
end
function initial_diffusion(diffusionmodel::FixedMVDiffusion, d, q, Eltype)
    initdiff = diffusionmodel.initial_diffusion
    @assert initdiff isa Number || length(initdiff) == d
    return kron(Diagonal(initdiff .* ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
end
estimate_local_diffusion(kind::FixedMVDiffusion, integ) = local_diagonal_diffusion(integ)
function estimate_global_diffusion(kind::FixedMVDiffusion, integ)
    @unpack q, measurement = integ.cache

    v, S = measurement.μ, measurement.Σ
    S_11 = diag(S)[1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(Σ, I(q + 1))

    if integ.success_iter == 0
        # @assert length(diffusions) == 0
        return Σ_out
    else
        # @assert length(diffusions) == integ.success_iter
        diffusion_prev = integ.cache.global_diffusion
        diffusion = diffusion_prev + (Σ_out - diffusion_prev) / integ.success_iter
        return diffusion
    end
end

"""
    local_scalar_diffusion(integ)

Computes a scalar local diffusion estimate:
```math
σ² = zᵀ ⋅ (H*Q*H')⁻¹ ⋅ z
```
"""
function local_scalar_diffusion(integ)
    @unpack d, R, H, Qh, measurement, m_tmp = integ.cache
    z = measurement.μ
    e, HQH = m_tmp.μ, m_tmp.Σ
    X_A_Xt!(HQH, Qh, H)
    if HQH.mat isa Diagonal
        e .= z ./ HQH.mat.diag
        σ² = dot(e, z) / d
        return σ²
    else
        C = cholesky!(HQH.mat)
        ldiv!(e, C, z)
        σ² = dot(z, e) / d
        return σ²
    end
end

"""
    local_diagonal_diffusion(integ)

Computes a diagonal local diffusion estimate:
```math
Σᵢᵢ = zᵢ² / (H*Q*H')ᵢᵢ
```
"""
function local_diagonal_diffusion(integ)
    @unpack q, H, Qh, measurement, m_tmp = integ.cache
    z = measurement.μ
    HQH = X_A_Xt!(m_tmp.Σ, Qh, H)
    Q0_11 = diag(HQH)[1]

    Σ_ii = z .^ 2 ./ Q0_11
    # Σ_ii .= max.(Σ_ii, eps(eltype(Σ_ii)))
    Σ = Diagonal(Σ_ii)

    Σ_out = kron(Σ, I(q + 1))
    return Σ_out
end
