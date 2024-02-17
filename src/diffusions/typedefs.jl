abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true

estimate_global_diffusion(diffusion::AbstractDynamicDiffusion, d, q, Eltype) =
    error("Not possible or not implemented")

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
estimate_local_diffusion(::FixedMVDiffusion, integ) =
    integ.alg isa EK0 ? local_diagonal_diffusion(integ.cache) :
    local_scalar_diffusion(integ.cache)
