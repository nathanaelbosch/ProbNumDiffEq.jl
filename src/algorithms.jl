########################################################################################
# Algorithm
########################################################################################
abstract type GaussianODEFilter <: DiffEqBase.AbstractODEAlgorithm end
abstract type AbstractEKF <: GaussianODEFilter end

struct EKF0 <: AbstractEKF
    prior::Symbol
    order::Int
    diffusionmodel::Symbol
    smooth::Bool
end
EKF0(; prior=:ibm, order=1, diffusionmodel=:dynamic, smooth=true) =
    EKF0(prior, order, diffusionmodel, smooth)

struct EKF1 <: AbstractEKF
    prior::Symbol
    order::Int
    diffusionmodel::Symbol
    smooth::Bool
end
EKF1(; prior=:ibm, order=1, diffusionmodel=:dynamic, smooth=true) =
    EKF1(prior, order, diffusionmodel, smooth)
