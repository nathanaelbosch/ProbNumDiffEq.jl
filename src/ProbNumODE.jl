__precompile__()

module ProbNumODE

using Reexport
@reexport using DiffEqBase
using OrdinaryDiffEq
import DiffEqBase: check_error!

import Base: copy, copy!, show, size
stack(x) = copy(reduce(hcat, x)')

using LinearAlgebra
using ForwardDiff
@reexport using StructArrays
using UnPack
using RecipesBase
using Distributions

@reexport using GaussianDistributions
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); nothing)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T, S}) where {T, S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)

using ModelingToolkit
using PDMats

import Statistics: mean, var, std
mean(s::StructArray{Gaussian{T,S}}) where {T,S} = mean.(s)
var(s::StructArray{Gaussian{T,S}}) where {T,S} = var.(s)
std(s::StructArray{Gaussian{T,S}}) where {T,S} = std.(s)
mean(s::DiffEqBase.DiffEqArray{A1, A2, StructArray{Gaussian{T,S}}, A3}) where {A1,A2,A3,T,S} =
     DiffEqBase.DiffEqArray(mean.(s.u), s.t)
var(s::DiffEqBase.DiffEqArray{A1, A2, StructArray{Gaussian{T,S}}, A3}) where {A1,A2,A3,T,S} =
    DiffEqBase.DiffEqArray(var.(s.u), s.t)
std(s::DiffEqBase.DiffEqArray{A1, A2, StructArray{Gaussian{T,S}}, A3}) where {A1,A2,A3,T,S} =
    DiffEqBase.DiffEqArray(std.(s.u), s.t)

include("steprules.jl")
include("priors.jl")
# include("measurement_model.jl")
include("diffusions.jl")

include("algorithms.jl")
export EKF0, EKF1
include("alg_utils.jl")
include("caches.jl")
include("state_initialization.jl")
include("integrator_utils.jl")
include("filtering.jl")
include("perform_step.jl")
include("preconditioning.jl")
include("smoothing.jl")
include("solution.jl")

# Utils
include("jacobian.jl")
include("numerics_tricks.jl")

end
