########################################################################################
# Algorithm
########################################################################################
abstract type GaussianODEFilter <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm end
abstract type AbstractEK <: GaussianODEFilter end

"""
    EK0(; order=3, smooth=true,
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit())

**Gaussian ODE filter with zeroth order vector field linearization**

- `order`: Order of the integrated Brownian motion (IBM) prior.
- `smooth`: Turn smoothing on/off; smoothing is required for dense output.
- `diffusionmodel`: See [Diffusion models and calibration](@ref).
- `initialization`: See [Initialization](@ref).

## [References](@ref references)
"""
Base.@kwdef struct EK0{DT,IT} <: AbstractEK
    order::Int = 3
    diffusionmodel::DT = DynamicDiffusion()
    smooth::Bool = true
    initialization::IT = TaylorModeInit()
end

"""
    EK1(; order=3, smooth=true,
          diffusionmodel=DynamicDiffusion(),
          initialization=TaylorModeInit(),
          kwargs...)

**Gaussian ODE filter with first order vector field linearization**

- `order`: Order of the integrated Brownian motion (IBM) prior.
- `smooth`: Turn smoothing on/off; smoothing is required for dense output.
- `diffusionmodel`: See [Diffusion models and calibration](@ref).
- `initialization`: See [Initialization](@ref).

Some additional `kwargs` relating to implicit solvers are supported;
check out DifferentialEquations.jl's [Extra Options](https://diffeq.sciml.ai/stable/solvers/ode_solve/#Extra-Options) page.
Right now, we support `autodiff`, `chunk_size`, and `diff_type`.
In particular, `autodiff=false` can come in handy to use finite differences instead of
ForwardDiff.jl to compute Jacobians.

## [References](@ref references)
"""
struct EK1{CS,AD,DiffType,DT,IT} <: AbstractEK
    order::Int
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
end
EK1(;
    order=3,
    diffusionmodel=DynamicDiffusion(),
    smooth=true,
    initialization=TaylorModeInit(),
    chunk_size=0,
    autodiff=true,
    diff_type=Val{:forward},
) = EK1{chunk_size,autodiff,diff_type,typeof(diffusionmodel),typeof(initialization)}(
    order,
    diffusionmodel,
    smooth,
    initialization,
)

Base.@kwdef struct EK1FDB{DT,IT} <: AbstractEK
    order::Int = 3
    diffusionmodel::DT = DynamicDiffusion()
    smooth::Bool = true
    initialization::IT = TaylorModeInit()
    jac_quality::Int = 1
end
