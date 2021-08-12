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

import Base: copy, copy!, show, size, ndims, similar
stack(x) = copy(reduce(hcat, x)')

using LinearAlgebra
import LinearAlgebra: mul!
# patch diagonal matrices:
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::Diagonal) =
    (C .= A .* B.diag')
using TaylorSeries
@reexport using StructArrays
using UnPack
using RecipesBase
using ModelingToolkit
using RecursiveArrayTools
using StaticArrays
using ForwardDiff
using Tullio
import Octavian: matmul!
# Define some fallbacks
_matmul!(C::AbstractVecOrMat{T}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T}) where {
    T <: Union{Bool, Float16, Float32, Float64,
               Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8}} =
                   matmul!(C, A, B)
_matmul!(C::AbstractVecOrMat{T}, A::AbstractVecOrMat{T}, B::AbstractVecOrMat{T},
         a::T, b::T) where {T <: Union{
             Bool, Float16, Float32, Float64,
             Int16, Int32, Int64, Int8, UInt16, UInt32, UInt64, UInt8}} =
                 matmul!(C, A, B, a, b)
_matmul!(C, A, B) = mul!(C, A, B)
_matmul!(C, A, B, a, b) = mul!(C, A, B, a, b)

matmul!(C::AbstractMatrix, A::AbstractMatrix, B::Diagonal) = (C .= A .* B.diag')
matmul!(C::AbstractMatrix, A::Diagonal, B::AbstractMatrix) = (C .= A.diag .* B)
matmul!(C::AbstractMatrix, A::AbstractMatrix, B::LowerTriangular) = mul!(C, A, B)
matmul!(C::Diagonal, A::AbstractMatrix, B::AbstractMatrix) = mul!(C, A, B)

matmul!(C::RecursiveArrayTools.ArrayPartition, A::AbstractMatrix, B::AbstractVector) = mul!(C, A, B)


# @reexport using PSDMatrices
# import PSDMatrices: X_A_Xt
include("squarerootmatrix.jl")
const SRMatrix = SquarerootMatrix
export SRMatrix
X_A_Xt(A, X) = X*A*X'
X_A_Xt!(out, A, X) = (out .= X*A*X')
apply_diffusion(Q, diffusion::Diagonal) = X_A_Xt(Q, sqrt.(diffusion))
apply_diffusion(Q::SRMatrix, diffusion::Number) = SRMatrix(sqrt.(diffusion)*Q.squareroot)


# All the Gaussian things
@reexport using GaussianDistributions
using GaussianDistributions: logpdf

import Statistics: mean, var, std

include("gaussians.jl")

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
include("projection.jl")
include("smoothing.jl")

include("solution.jl")
include("solution_sampling.jl")
include("solution_plotting.jl")

include("preconditioning.jl")

# Utils
include("jacobian.jl")
include("numerics_tricks.jl")

# Iterated Extended Kalman Smoother
include("ieks.jl")
export IEKS, solve_ieks

end
