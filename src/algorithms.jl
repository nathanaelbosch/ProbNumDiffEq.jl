########################################################################################
# Algorithm
########################################################################################
abstract type GaussianODEFilter <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm end
abstract type AbstractEKF <: GaussianODEFilter end

"""
    EKF0(; prior=:ibm, order=1, diffusionmodel=:dynamic, smooth=true)

Gaussian ODE filtering with zeroth order extended Kalman filter.

Currently, only the integrated Brownian motion prior `:ibm` is supported.
For the diffusionmodel, chose one of
`[:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP]`.

See also: [`EKF1`](@ref)

# References:
- M. Schober, S. Särkkä, and P. Hennig: **A Probabilistic Model for the Numerical Solution
  of Initial Value Problems**
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary
  Differential Equations As Non-Linear Bayesian Filtering: A New Perspective**
"""
Base.@kwdef struct EKF0 <: AbstractEKF
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


"""
    EKF1(; prior=:ibm, order=1, diffusionmodel=:dynamic, smooth=true)

Gaussian ODE filtering with first order extended Kalman filter

Currently, only the integrated Brownian motion prior `:ibm` is supported.
For the diffusionmodel, chose one of
`[:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP]`.

See also: [`EKF0`](@ref)

# References:
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective**
"""
Base.@kwdef struct EKF1 <: AbstractEKF
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end
