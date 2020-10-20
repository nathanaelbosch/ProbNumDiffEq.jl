__precompile__()

module ProbNumODE

using Reexport
@reexport using DiffEqBase
import DiffEqBase: check_error!

import Base: copy, copy!

using LinearAlgebra
using Measurements
using ForwardDiff
@reexport using StructArrays
using UnPack
using StaticArrays
using RecipesBase
using Distributions
@reexport using GaussianDistributions
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); nothing)
using ModelingToolkit
using DiffEqDevTools
using Optim
using PDMats

using UUIDs, ProgressLogging

import Base: copy

stack(x) = copy(reduce(hcat, x)')

include("steprules.jl")
include("priors.jl")
# include("measurement_model.jl")
include("sigmas.jl")
include("error_estimation.jl")

include("integrator_type.jl")
export EKF0, EKF1, ODEFilter
include("integrator_interface.jl")
include("integrator_utils.jl")
include("filtering.jl")
include("perform_step.jl")
include("solve.jl")
include("preconditioning.jl")
include("calibration.jl")
include("smoothing.jl")
include("solution.jl")

include("dev/problems.jl")
export exponential_decay, logistic_equation, brusselator, fitzhugh_nagumo, lotka_volterra, van_der_pol, fitzhugh_nagumo_iip
include("dev/evaluation.jl")
export MyWorkPrecision, MyWorkPrecisionSet, plot_wps

include("utils/progressbar.jl")
include("utils/rhs_derivatives.jl")
include("utils/numerics_tricks.jl")

end
