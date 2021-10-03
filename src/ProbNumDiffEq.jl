__precompile__()

module ProbNumDiffEq

using Reexport
@reexport using DiffEqBase
import DiffEqBase: check_error!, AbstractODEFunction
using OrdinaryDiffEq
using DiffEqDevTools

# Current working solution to depending on functions that moved from DiffEqBase to SciMLBase
try
    DiffEqBase.interpret_vars
    DiffEqBase.getsyms # these are here to trigger an error
    import DiffEqBase: interpret_vars, getsyms
catch
    import DiffEqBase.SciMLBase: interpret_vars, getsyms
end

import Base: copy, copy!, show, size, ndims, similar
stack(x) = copy(reduce(hcat, x)')

using LinearAlgebra
import LinearAlgebra: mul!
# patch diagonal matrices:
LinearAlgebra.mul!(C::AbstractMatrix, A::AbstractMatrix, B::Diagonal) = (C .= A .* B.diag')
using TaylorSeries
using TaylorIntegration
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
const OctavianCompatibleEltypes =
    Union{Bool,Float16,Float32,Float64,Int16,Int32,Int64,Int8,UInt16,UInt32,UInt64,UInt8}
_matmul!(
    C::AbstractVecOrMat{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
) where {T<:OctavianCompatibleEltypes} = matmul!(C, A, B)
_matmul!(
    C::AbstractVecOrMat{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
    a::T,
    b::T,
) where {T<:OctavianCompatibleEltypes} = matmul!(C, A, B, a, b)
_matmul!(C, A, B) = mul!(C, A, B)
_matmul!(C, A, B, a, b) = mul!(C, A, B, a, b)
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::Diagonal{T},
) where {T<:OctavianCompatibleEltypes} = (C .= A .* B.diag')
_matmul!(
    C::AbstractMatrix{T},
    A::Diagonal{T},
    B::AbstractMatrix{T},
) where {T<:OctavianCompatibleEltypes} = (C .= A.diag .* B)
_matmul!(
    C::Diagonal{T},
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
) where {T<:OctavianCompatibleEltypes} = @tullio C[i, i] = A[i, j] * B[j, i]

# @reexport using PSDMatrices
# import PSDMatrices: X_A_Xt
include("squarerootmatrix.jl")
const SRMatrix = SquarerootMatrix
export SRMatrix
X_A_Xt(A, X) = X * A * X'
X_A_Xt!(out, A, X) = (out .= X * A * X')
apply_diffusion(Q, diffusion::Diagonal) = X_A_Xt(Q, sqrt.(diffusion))
apply_diffusion(Q::SRMatrix, diffusion::Number) = SRMatrix(sqrt.(diffusion) * Q.squareroot)

# All the Gaussian things
@reexport using GaussianDistributions
using GaussianDistributions: logpdf

import Statistics: mean, var, std

include("gaussians.jl")

include("priors.jl")
include("diffusions.jl")

include("initialization/common.jl")
export TaylorModeInit, RungeKuttaInit

include("algorithms.jl")
export EK0, EK1, EK1FDB

include("alg_utils.jl")
include("caches.jl")

include("initialization/taylormode.jl")
include("initialization/rungekutta.jl")

include("integrator_utils.jl")
include("filtering/predict.jl")
include("filtering/update.jl")
include("filtering/smooth.jl")
include("perform_step.jl")
include("projection.jl")

include("solution.jl")
include("solution_sampling.jl")
include("solution_plotting.jl")

include("preconditioning.jl")

# Utils
include("jacobian.jl")

# Iterated Extended Kalman Smoother
include("ieks.jl")
export IEKS, solve_ieks

include("devtools.jl")
include("callbacks.jl")
export ManifoldUpdate

# Do as they do here:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/v5.61.1/src/OrdinaryDiffEq.jl#L175-L193
let
    while true
        function lorenz(du, u, p, t)
            du[1] = 10.0(u[2] - u[1])
            du[2] = u[1] * (28.0 - u[3]) - u[2]
            return du[3] = u[1] * u[2] - (8 / 3) * u[3]
        end

        lorenzprob = ODEProblem(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0))
        solve(lorenzprob, EK0())
        solve(lorenzprob, EK1())
        break
    end
end

end
