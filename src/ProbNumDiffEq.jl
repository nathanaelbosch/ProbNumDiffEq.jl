__precompile__()

module ProbNumDiffEq

import Base: copy, copy!, show, size, ndims, similar, isapprox, isequal, iterate, ==, length

using LinearAlgebra
import LinearAlgebra: mul!
import Statistics: mean, var, std

using Reexport
@reexport using DiffEqBase
import SciMLBase
import SciMLBase: interpret_vars, getsyms, remake
using OrdinaryDiffEq
using SpecialMatrices, ToeplitzMatrices
using FastBroadcast
using StaticArrayInterface
using FunctionWrappersWrappers
using TaylorSeries, TaylorIntegration
@reexport using StructArrays
using SimpleUnPack
using RecursiveArrayTools
using ForwardDiff
using ExponentialUtilities
using Octavian
using FastGaussQuadrature
import Kronecker
using ArrayAllocators
using FiniteHorizonGramians
using FillArrays

@reexport using GaussianDistributions

@reexport using PSDMatrices
import PSDMatrices: X_A_Xt, X_A_Xt!, unfactorize
X_A_Xt(A, X) = X * A * X'

stack(x) = copy(reduce(hcat, x)')
vecvec2mat(x) = reduce(hcat, x)'

include("fast_linalg.jl")
include("kronecker.jl")
include("covariance_structure.jl")

abstract type AbstractODEFilterCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end

include("gaussians.jl")

include("priors/common.jl")
include("priors/iwp.jl")
include("priors/ltisde.jl")
include("priors/ioup.jl")
include("priors/matern.jl")
export IWP, IOUP, Matern
include("diffusions.jl")
export FixedDiffusion, DynamicDiffusion, FixedMVDiffusion, DynamicMVDiffusion

include("initialization/common.jl")
export TaylorModeInit, ClassicSolverInit, SimpleInit, ForwardDiffInit

include("algorithms.jl")
export EK0, EK1
export ExpEK, RosenbrockExpEK

include("alg_utils.jl")
include("caches.jl")

include("checks.jl")

include("initialization/simpleinit.jl")
include("initialization/autodiffinit.jl")
include("initialization/classicsolverinit.jl")

include("solution.jl")
include("solution_sampling.jl")
# include("solution_plotting.jl")

include("integrator_utils.jl")
include("filtering/markov_kernel.jl")
include("filtering/predict.jl")
include("filtering/update.jl")
include("filtering/smooth.jl")
include("measurement_models.jl")
include("derivative_utils.jl")
include("perform_step.jl")
include("projection.jl")
include("solve.jl")

include("preconditioning.jl")

if !isdefined(Base, :get_extension)
    include("../ext/DiffEqDevToolsExt.jl")
end

include("callbacks.jl")
export ManifoldUpdate

include("precompile.jl")

end
