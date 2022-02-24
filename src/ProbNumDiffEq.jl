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
using TaylorSeries
using TaylorIntegration
@reexport using StructArrays
using UnPack
using RecipesBase
using RecursiveArrayTools
using StaticArrays
using ForwardDiff
using Tullio
# By default use mul!
_matmul!(C, A, B) = mul!(C, A, B)
_matmul!(C, A, B, a, b) = mul!(C, A, B, a, b)
# Some special cases
_matmul!(C::AbstractMatrix{Float64}, A::AbstractMatrix{Float64}, B::Diagonal{Float64}) =
    (C .= A .* B.diag')
_matmul!(C::AbstractMatrix{Float64}, A::Diagonal{Float64}, B::AbstractMatrix{Float64}) =
    (C .= A.diag .* B)
_matmul!(C::Diagonal{Float64}, A::AbstractMatrix{Float64}, B::AbstractMatrix{Float64}) =
    @tullio C[i, i] = A[i, j] * B[j, i]
# Use Octavian's matmul! when the types are very specific
import Octavian: matmul!
_matmul!(
    C::AbstractVecOrMat{Float64},
    A::AbstractVecOrMat{Float64},
    B::AbstractVecOrMat{Float64},
) = matmul!(C, A, B)
_matmul!(
    C::AbstractVecOrMat{Float64},
    A::AbstractVecOrMat{Float64},
    B::AbstractVecOrMat{Float64},
    a::Float64,
    b::Float64,
) = matmul!(C, A, B, a, b)

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
export FixedDiffusion, DynamicDiffusion, FixedMVDiffusion, DynamicMVDiffusion

include("initialization/common.jl")
export TaylorModeInit, ClassicSolverInit

include("algorithms.jl")
export EK0, EK1, EK1FDB

include("alg_utils.jl")
include("caches.jl")

include("initialization/taylormode.jl")
include("initialization/classicsolverinit.jl")

include("solution.jl")
include("solution_sampling.jl")
include("solution_plotting.jl")

include("integrator_utils.jl")
include("filtering/predict.jl")
include("filtering/update.jl")
include("filtering/smooth.jl")
include("perform_step.jl")
include("projection.jl")
include("solve.jl")

include("preconditioning.jl")

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
