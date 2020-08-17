__precompile__()

module ProbNumODE

using Reexport
@reexport using DiffEqBase

import Base: copy

using LinearAlgebra
using Measurements
using ForwardDiff
using StructArrays
using UnPack
using StaticArrays
using RecipesBase
using Distributions
using GaussianDistributions
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
using ModelingToolkit

using UUIDs, ProgressLogging

import Base: copy

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)
stack(x) = copy(reduce(hcat, x)')

include("filtering.jl")
include("steprules.jl")
include("priors.jl")
include("measurement_model.jl")
include("sigmas.jl")

include("integrator_type.jl")
export EKF0, EKF1, ODEFilter
include("integrator_interface.jl")
include("integrator_utils.jl")
include("perform_step.jl")
include("solve.jl")
# include("preconditioning.jl")
include("postprocessing.jl")
include("solution.jl")

include("dev/problems.jl")
export exponential_decay, logistic_equation, brusselator, fitzhugh_nagumo, lotka_volterra, van_der_pol, fitzhugh_nagumo_iip
include("dev/visualization.jl")
export hairer_plot

include("utils/progressbar.jl")
include("utils/rhs_derivatives.jl")

end
