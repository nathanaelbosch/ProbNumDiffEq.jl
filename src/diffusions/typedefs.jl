abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true

"""
    DynamicDiffusion()

Time-varying, isotropic diffusion, which is quasi-maximum-likelihood-estimated at each step.

**This is the recommended diffusion when using adaptive step-size selection,** and in
particular also when solving stiff systems.
"""
struct DynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicDiffusion, d, Eltype) = one(Eltype)
estimate_local_diffusion(::DynamicDiffusion, integ, obs, HQH) =
    local_scalar_diffusion(integ.cache, obs.r, HQH)

"""
    DynamicMVDiffusion()

Time-varying, diagonal diffusion, which is quasi-maximum-likelihood-estimated at each step.

**Supported by the [`EK0`](@ref) with the [`IWP`](@ref) prior (the default) and by the
[`DiagonalEK1`](@ref).** Other setups throw an `ArgumentError`.

A multi-variate version of [`DynamicDiffusion`](@ref), where instead of an isotropic matrix,
a diagonal matrix is estimated. This can be helpful to get more expressive posterior
covariances when using the [`EK0`](@ref), since the individual dimensions can be adjusted
separately.

# References
* [Bosch et al. (2021)](@cite bosch20capos) "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS
"""
struct DynamicMVDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(::DynamicMVDiffusion, d, Eltype) = Diagonal(ones(Eltype, d))
estimate_local_diffusion(::DynamicMVDiffusion, integ, obs, HQH) =
    local_diagonal_diffusion(integ.cache, obs.r, HQH)

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
initial_diffusion(diffusionmodel::FixedDiffusion, d, Eltype) =
    diffusionmodel.initial_diffusion * one(Eltype)
estimate_local_diffusion(::FixedDiffusion, integ, obs, HQH) =
    local_scalar_diffusion(integ.cache, obs.r, HQH)

"""
    FixedMVDiffusion(; initial_diffusion=1.0, calibrate=true)

Time-fixed, diagonal diffusion. With `calibrate=true` it is
quasi-maximum-likelihood-estimated from the whole solve.

**With `calibrate=true`, supported by the [`EK0`](@ref) with the [`IWP`](@ref) prior (the
default) and by the [`DiagonalEK1`](@ref).** With `calibrate=false`, the given
`initial_diffusion` is used as is, and every solver supports it. The setups that support
`calibrate=true` also use a per-dimension diffusion in the local error estimate for adaptive
step size selection; all other setups use a scalar estimate there.

A multi-variate version of [`FixedDiffusion`](@ref), where instead of an isotropic matrix,
a diagonal matrix is estimated. This can be helpful to get more expressive posterior
covariances when using the [`EK0`](@ref), since the individual dimensions can be adjusted
separately.

# References
* [Bosch et al. (2021)](@cite bosch20capos) "Calibrated Adaptive Probabilistic ODE Solvers", AISTATS
"""
Base.@kwdef struct FixedMVDiffusion{T} <: AbstractStaticDiffusion
    initial_diffusion::T = 1.0
    calibrate::Bool = true
end
function initial_diffusion(diffusionmodel::FixedMVDiffusion, d, Eltype)
    initdiff = diffusionmodel.initial_diffusion
    if initdiff isa Number
        return initdiff * one(Eltype) * I(d)
    elseif initdiff isa AbstractVector
        @assert length(initdiff) == d
        return Diagonal(initdiff)
    elseif initdiff isa Diagonal
        @assert size(initdiff) == (d, d)
        return initdiff
    else
        throw(
            ArgumentError(
                "Invalid `initial_diffusion`. The `FixedMVDiffusion` assumes a dxd diagonal diffusion model. So, pass either a Number, a Vector of length d, or a `Diagonal`.",
            ),
        )
    end
end
# A block-diagonal `H Q Hᵀ` comes from an `H` that does not couple the dimensions. It is then
# diagonal, and the estimate of one diffusion per dimension is exact.
estimate_local_diffusion(::FixedMVDiffusion, integ, obs, HQH::BlocksOfDiagonalsPSD) =
    local_diagonal_diffusion(integ.cache, obs.r, HQH)
# Otherwise, the scalar estimate is written into every entry, since the local diffusion of a
# multivariate model is stored as a `Diagonal`
function estimate_local_diffusion(::FixedMVDiffusion, integ, obs, HQH)
    σ² = local_scalar_diffusion(integ.cache, obs.r, HQH)
    fill!(integ.cache.local_diffusion.diag, σ²)
    return integ.cache.local_diffusion
end
