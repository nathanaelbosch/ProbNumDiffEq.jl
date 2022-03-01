########################################################################################
# Algorithm
########################################################################################
abstract type GaussianODEFilter <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm end
abstract type AbstractEK <: GaussianODEFilter end

"""
    EK0(; order=3, diffusionmodel=DynamicDiffusion(), smooth=true)

**Gaussian ODE filtering with zeroth order extended Kalman filtering.**

All solvers use an integrated Brownian motion prior of order `order`.
For the diffusionmodel, chose one of
`[DynamicDiffusion(), DynamicMVDiffusion(), FixedDiffusion(), FixedMVDiffusion()]`.

See also: [`EK1`](@ref)

# References:
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019)
- M. Schober, S. Särkkä, and P. Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems** (2018)
"""
Base.@kwdef struct EK0{DT,IT} <: AbstractEK
    order::Int = 3
    diffusionmodel::DT = DynamicDiffusion()
    smooth::Bool = true
    initialization::IT = TaylorModeInit()
end

"""
    EK1(; order=3, diffusionmodel=DynamicDiffusion(), smooth=true)

**Gaussian ODE filtering with first order extended Kalman filtering.**

All solvers use an integrated Brownian motion prior of order `order`.
For the diffusionmodel, chose one of
`[DynamicDiffusion(), DynamicMVDiffusion(), FixedDiffusion(), FixedMVDiffusion()]`.

See also: [`EK0`](@ref)

# References:
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019)
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
