########################################################################################
# Algorithm
########################################################################################
abstract type GaussianODEFilter <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm end
abstract type AbstractEK <: GaussianODEFilter end

"""
    EK0(; prior=:ibm, order=3, diffusionmodel=:dynamic, smooth=true)

**Gaussian ODE filtering with zeroth order extended Kalman filtering.**

Currently, only the integrated Brownian motion prior `:ibm` is supported.
For the diffusionmodel, chose one of
`[:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP]`.

See also: [`EK1`](@ref)

# References:
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019)
- M. Schober, S. Särkkä, and P. Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems** (2018)
"""
Base.@kwdef struct EK0{IT} <: AbstractEK
    prior::Symbol = :ibm
    order::Int = 3
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
    initialization::IT = TaylorModeInit()
end

"""
    EK1(; prior=:ibm, order=3, diffusionmodel=:dynamic, smooth=true)

**Gaussian ODE filtering with first order extended Kalman filtering.**

Currently, only the integrated Brownian motion prior `:ibm` is supported.
For the diffusionmodel, chose one of
`[:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP]`.

See also: [`EK0`](@ref)

# References:
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019)
"""
struct EK1{CS,AD,DiffType,IT} <: AbstractEK
    prior::Symbol
    order::Int
    diffusionmodel::Symbol
    smooth::Bool
    initialization::IT
end
EK1(;
    prior=:ibm,
    order=3,
    diffusionmodel=:dynamic,
    smooth=true,
    initialization=TaylorModeInit(),
    chunk_size=0,
    autodiff=true,
    diff_type=Val{:forward},
) = EK1{chunk_size,autodiff,diff_type,typeof(initialization)}(
    prior,
    order,
    diffusionmodel,
    smooth,
    initialization,
)

Base.@kwdef struct EK1FDB{IT} <: AbstractEK
    prior::Symbol = :ibm
    order::Int = 3
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
    initialization::IT = TaylorModeInit()
    jac_quality::Int = 1
end
