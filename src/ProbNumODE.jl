__precompile__()

module ProbNumODE

using LinearAlgebra
using Measurements
using Distributions
using ForwardDiff
using StructArrays
using Reexport
@reexport using DiffEqBase
using UnPack
using Plots
using StaticArrays

@inline _copy(a::SArray) = a
@inline _copy(a) = copy(a)

include("filtering.jl")
include("steprules.jl")
include("visualization.jl")
export hairer_plot
include("utils.jl")
include("sigmas.jl")
include("algorithm.jl")
export prob_solve


include("problems.jl")
export exponential_decay, logistic_equation, brusselator, fitzhugh_nagumo, lotka_volterra, van_der_pol

include("diffeq.jl")

end
