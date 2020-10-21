########################################################################################
# Algorithm
########################################################################################
abstract type GaussianODEFilter <: DiffEqBase.AbstractODEAlgorithm end
abstract type AbstractEKF <: GaussianODEFilter end

Base.@kwdef struct EKF0 <: AbstractEKF
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end

Base.@kwdef struct EKF1 <: AbstractEKF
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end
