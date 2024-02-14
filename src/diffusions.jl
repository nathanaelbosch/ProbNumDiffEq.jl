abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true

apply_diffusion(Q::PSDMatrix{T, <:Matrix}, diffusion::Diagonal) where {T} = begin
    d = size(diffusion, 1)
    q = size(Q, 1) ÷ d - 1
    return PSDMatrix(Q.R * sqrt.(kron(diffusion, I(q+1))))
end
apply_diffusion(
    Q::PSDMatrix{T, <:IsometricKroneckerProduct},
    diffusion::Diagonal{T, <:FillArrays.Fill},
) where {T} = begin
    PSDMatrix(Q.R * sqrt.(diffusion.diag.value))
end
apply_diffusion(Q::PSDMatrix{T, <:BlockDiagonal}, diffusion::Diagonal) where {T} = begin
    PSDMatrix(BlockDiagonal([
        Q.R.blocks[i] * sqrt.(diffusion.diag[i]) for i in eachindex(Q.R.blocks)
    ]))
end

apply_diffusion!(Q::PSDMatrix, diffusion::Diagonal{T, <:FillArrays.Fill}) where {T} =
    rmul!(Q.R, sqrt.(diffusion.diag.value))
apply_diffusion!(
    Q::PSDMatrix{T,<:BlockDiagonal},
    diffusion::Diagonal{T,<:Vector},
) where {T} =
    @simd ivdep for i in eachindex(blocks(Q.R))
        rmul!(blocks(Q.R)[i], diffusion.diag[i])
    end

apply_diffusion!(out::PSDMatrix, Q::PSDMatrix, diffusion::Diagonal{T,<:FillArrays.Fill}) where {T} =
    rmul!(Q.R, sqrt.(diffusion.diag.value))


estimate_global_diffusion(diffusion::AbstractDynamicDiffusion, d, q, Eltype) = NaN

"""
    DynamicDiffusion()

Time-varying, isotropic diffusion, which is quasi-maximum-likelihood-estimated at each step.

**This is the recommended diffusion when using adaptive step-size selection,** and in
particular also when solving stiff systems.
"""
struct DynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicDiffusion, d, q, Eltype) = one(Eltype) * Eye(d)
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
initial_diffusion(::DynamicMVDiffusion, d, q, Eltype) = Diagonal(ones(Eltype, d))
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
    diffusionmodel.initial_diffusion * one(Eltype) * Eye(d)
estimate_local_diffusion(::FixedDiffusion, integ) = local_scalar_diffusion(integ.cache)
function estimate_global_diffusion(::FixedDiffusion, integ)
    @unpack d, measurement, m_tmp, Smat = integ.cache
    # sol_diffusions = integ.sol.diffusions

    v, S = measurement.μ, measurement.Σ
    e = m_tmp.μ
    e .= v
    diffusion_t = if S isa IsometricKroneckerProduct
        @assert length(S.B) == 1
        dot(v, e) / d / S.B[1]
    elseif S isa BlockDiagonal
        @assert length(S.blocks) == d
        @assert length(S.blocks[1]) == 1
        @simd ivdep for i in eachindex(e)
            @inbounds e[i] /= S.blocks[i][1]
        end
        dot(v, e) / d
    else
        S_chol = cholesky!(copy!(Smat, S))
        ldiv!(S_chol, e)
        dot(v, e) / d
    end

    if integ.success_iter == 0
        # @assert length(sol_diffusions) == 0
        global_diffusion = diffusion_t
        integ.cache.global_diffusion = global_diffusion * Eye(d)
        return integ.cache.global_diffusion
    else
        # @assert length(sol_diffusions) == integ.success_iter
        diffusion_prev = integ.cache.global_diffusion.diag.value
        global_diffusion =
            diffusion_prev + (diffusion_t - diffusion_prev) / integ.success_iter
        # @info "compute diffusion" diffusion_prev global_diffusion
        integ.cache.global_diffusion = global_diffusion * Eye(d)
        return integ.cache.global_diffusion
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
    return Diagonal(initdiff .* ones(Eltype, d))
end
estimate_local_diffusion(::FixedMVDiffusion, integ) = local_diagonal_diffusion(integ.cache)
function estimate_global_diffusion(::FixedMVDiffusion, integ)
    @unpack d, q, measurement, local_diffusion = integ.cache

    v, S = measurement.μ, measurement.Σ
    # @assert diag(S) |> unique |> length == 1
    S_11 = S[1, 1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(Σ, I(q + 1)) # -> Different for each dimension; same for each derivative

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
    _matmul!(HQH, C_Dxd', C_Dxd)
    e .= z
    σ² = if HQH isa IsometricKroneckerProduct
        @assert length(HQH.B) == 1
        dot(z, e) / d / HQH.B[1]
    elseif HQH isa BlockDiagonal
        @assert length(HQH.blocks) == d
        @assert length(HQH.blocks[1]) == 1
        for i in eachindex(e)
            e[i] /= HQH.blocks[i][1]
        end
        dot(z, e) / d
    else
        C = cholesky!(HQH)
        ldiv!(C, e)
        dot(z, e) / d
    end
    cache.local_diffusion = σ² * Eye(d)
    return cache.local_diffusion
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
    # HQH = H * unfactorize(Qh) * H'
    # @assert HQH |> diag |> unique |> length == 1
    # c1 = view(_matmul!(cache.C_Dxd, Qh.R, H'), :, 1)
    Q0_11 = if Qh.R isa BlockDiagonal
        c1 = mul!(view(cache.C_Dxd.blocks[1], :, 1:1), Qh.R.blocks[1], view(H.blocks[1], 1:1, :)')
        dot(c1, c1)
    else
        c1 = mul!(view(cache.C_Dxd, :, 1:1), Qh.R, view(H, 1:1, :)')
        dot(c1, c1)
    end

    @. local_diffusion.diag = z^2 / Q0_11
    return local_diffusion
end
