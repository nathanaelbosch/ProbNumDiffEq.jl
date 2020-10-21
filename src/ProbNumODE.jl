__precompile__()

module ProbNumODE

using Reexport
@reexport using DiffEqBase
using OrdinaryDiffEq
import DiffEqBase: check_error!

import Base: copy, copy!

using LinearAlgebra
using ForwardDiff
@reexport using StructArrays
using UnPack
using RecipesBase
using Distributions
@reexport using GaussianDistributions
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); nothing)
using ModelingToolkit
using PDMats

import Base: copy

stack(x) = copy(reduce(hcat, x)')

include("steprules.jl")
include("priors.jl")
# include("measurement_model.jl")
include("diffusions.jl")

include("algorithms.jl")
export EKF0, EKF1
include("alg_utils.jl")
include("caches.jl")
include("state_initialization.jl")
include("integrator_type.jl")
include("integrator_interface.jl")
include("integrator_utils.jl")
include("filtering.jl")
include("perform_step.jl")
include("solve.jl")
include("preconditioning.jl")
include("smoothing.jl")
include("solution.jl")

# Utils
include("jacobian.jl")
include("numerics_tricks.jl")

end
