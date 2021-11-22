abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true
initial_diffusion(diffusion::AbstractDiffusion, d, q, Eltype) = one(Eltype)

struct DynamicDiffusion <: AbstractDynamicDiffusion end
estimate_local_diffusion(kind::DynamicDiffusion, integ) = local_scalar_diffusion(integ)

struct DynamicMVDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(diffusionmodel::DynamicMVDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
estimate_local_diffusion(kind::DynamicMVDiffusion, integ) = local_diagonal_diffusion(integ)


"""
    FixedDiffusion(; initial_diffusion=1.0, calibrate=True)

Time-fixed diffusion model with or without calibration.
The initial diffusion can be set via `initial_diffusion`; it will be used for each
prediction during the solve.
With calibration, this model accumulates the quasi-MLE during the forward solve of the ODE
and the filtering states are then calibrated correspondingly.
Without calibration, the model has a constant diffusion of `initial_diffusion`.
In addition, a local diffusion estimate is computed as in [`DynamicDiffusion`](@ref), which
is used for local error estimation and step-size adaptation.
"""
Base.@kwdef struct FixedDiffusion{T} <: AbstractStaticDiffusion
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
end
initial_diffusion(diffusionmodel::FixedMVDiffusion, d, q, Eltype) =
    diffusionmodel.initial_diffusion .*
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
estimate_local_diffusion(kind::FixedMVDiffusion, integ) = local_diagonal_diffusion(integ)
function estimate_global_diffusion(kind::FixedMVDiffusion, integ)
    @unpack d, q, R, P, PI, E1 = integ.cache
    @unpack measurement, H = integ.cache
    @unpack d, measurement = integ.cache
    # diffusions = integ.sol.diffusions

    # Assert EK0
    @assert all(H .== E1)

    @unpack measurement = integ.cache
    v, S = measurement.μ, measurement.Σ

    # More safety checks
    @assert isdiag(S)
    # @assert length(unique(diag(S))) == 1
    S_11 = diag(S)[1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(Σ, I(q + 1))
    @assert isdiag(Σ_out)
    # @info "MV-MLE-Diffusion" v S Σ Σ_out

    if integ.success_iter == 0
        # @assert length(diffusions) == 0
        return Σ_out
    else
        @assert length(diffusions) == integ.success_iter
        diffusion_prev = integ.cache.global_diffusion
        diffusion = diffusion_prev + (Σ_out - diffusion_prev) / integ.success_iter
        return diffusion
    end
end


"""
Local scalar diffusion:
σ² = zᵀ ⋅ (H*Q*H')⁻¹ ⋅ z
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
Local diagonal diffusion:
Σᵢᵢ = zᵢ² / (H*Q*H')ᵢᵢ
"""
function local_diagonal_diffusion(integ)
    @unpack d, q, R, P, PI, E1 = integ.cache
    @unpack H, Qh, measurement, m_tmp = integ.cache
    z = measurement.μ

    # @assert all(R .== 0) "The dynamic-diffusion assumes R==0!"

    # Assert EK0
    @assert all(H .== E1)

    # More safety checks
    HQH = X_A_Xt!(m_tmp.Σ, Qh, H)
    @assert isdiag(HQH)
    @assert length(unique(diag(HQH))) == 1
    Q0_11 = diag(HQH)[1]

    Σ_ii = z .^ 2 ./ Q0_11
    Σ_ii .= max.(Σ_ii, eps(eltype(Σ_ii)))
    Σ = Diagonal(Σ_ii)

    Σ_out = kron(Σ, I(q + 1))
    @assert isdiag(Σ_out)
    # @info "DynamicMV diffusion" Σ Σ_out
    return Σ_out
end
