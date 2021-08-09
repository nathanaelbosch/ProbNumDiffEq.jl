__precompile__()

module ProbNumDiffEq

using Reexport
@reexport using DiffEqBase
import DiffEqBase: check_error!, AbstractODEFunction
using OrdinaryDiffEq

# Current working solution to depending on functions that moved from DiffEqBase to SciMLBase
try
    DiffEqBase.interpret_vars; DiffEqBase.getsyms; # these are here to trigger an error
    import DiffEqBase: interpret_vars, getsyms
catch
    import DiffEqBase.SciMLBase: interpret_vars, getsyms
end

import Base: copy, copy!, show, size, ndims
stack(x) = copy(reduce(hcat, x)')

using LinearAlgebra
using TaylorSeries
@reexport using StructArrays
using UnPack
using RecipesBase
using ModelingToolkit
using RecursiveArrayTools
using StaticArrays
using ForwardDiff
using Tullio


# @reexport using PSDMatrices
# import PSDMatrices: X_A_Xt
include("squarerootmatrix.jl")
const SRMatrix = SquarerootMatrix
export SRMatrix
X_A_Xt(A, X) = X*A*X'
apply_diffusion(Q, diffusion::Diagonal) = X_A_Xt(Q, sqrt.(diffusion))
apply_diffusion(Q::SRMatrix, diffusion::Number) = SRMatrix(sqrt.(diffusion)*Q.squareroot)


# All the Gaussian things
@reexport using GaussianDistributions
using GaussianDistributions: logpdf
# const PSDGaussian{T} = Gaussian{Vector{T}, PSDMatrix{T}}
# const PSDGaussianList{T} = StructArray{PSDGaussian{T}}
const SRGaussian{T, S, M} = Gaussian{Vector{T}, SRMatrix{T, S, M}}
const SRGaussianList{T, S, M} = StructArray{SRGaussian{T, S, M}}
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); dst)
RecursiveArrayTools.recursivecopy(P::Gaussian) = copy(P)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T, S}) where {T, S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)
ndims(g::Gaussian) = ndims(g.μ)

Base.:*(M, g::SRGaussian) = Gaussian(M * g.μ, X_A_Xt(g.Σ, M))
# GaussianDistributions.whiten(Σ::SRMatrix, z) = Σ.L\z

import Statistics: mean, var, std
var(p::SRGaussian{T}) where {T} = diag(p.Σ)
std(p::SRGaussian{T}) where {T} = sqrt.(diag(p.Σ))
mean(s::SRGaussianList{T}) where {T} = s.μ
var(s::SRGaussianList{T}) where {T} = diag.(s.Σ)
std(s::SRGaussianList{T}) where {T} = map(v -> sqrt.(v), (diag.(s.Σ)))

include("priors.jl")
include("diffusions.jl")

include("algorithms.jl")
export EK0, EK1
include("alg_utils.jl")
include("caches.jl")
include("state_initialization.jl")
include("integrator_utils.jl")
include("filtering.jl")
include("perform_step.jl")
include("preconditioning.jl")
include("projection.jl")
include("smoothing.jl")

include("solution.jl")
include("solution_sampling.jl")
include("solution_plotting.jl")

# Utils
include("jacobian.jl")
include("numerics_tricks.jl")

# Iterated Extended Kalman Smoother
include("ieks.jl")
export IEKS, solve_ieks

end
