__precompile__()

module ProbNumODE

using Reexport
@reexport using DiffEqBase

using LinearAlgebra
using Measurements
using ForwardDiff
using StructArrays
using UnPack
using StaticArrays
using RecipesBase

using UUIDs, ProgressLogging

import Base: copy

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)

include("filtering.jl")
include("steprules.jl")
include("visualization.jl")
export hairer_plot
include("utils.jl")
include("sigmas.jl")
include("algorithm.jl")

include("problems.jl")
export exponential_decay, logistic_equation, brusselator, fitzhugh_nagumo, lotka_volterra, van_der_pol

include("diffeq.jl")

end
