__precompile__()

module ODEFilters

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
const MvNormal{T} = Gaussian{Vector{T}, Matrix{T}}
const MvNormalList{T} = StructArray{MvNormal{T}}
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); nothing)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T, S}) where {T, S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)

using ModelingToolkit
using PDMats

import Statistics: mean, var, std
var(p::MvNormal{T}) where {T} = diag(p.Σ)
std(p::MvNormal{T}) where {T} = sqrt.(diag(p.Σ))
mean(s::MvNormalList{T}) where {T} = mean.(s)
var(s::MvNormalList{T}) where {T} = var.(s)
std(s::MvNormalList{T}) where {T} = std.(s)

include("priors.jl")
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
