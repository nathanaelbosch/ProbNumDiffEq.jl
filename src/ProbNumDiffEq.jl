__precompile__()

module ProbNumDiffEq

using Reexport
@reexport using DiffEqBase
import DiffEqBase: check_error!, AbstractODEFunction
using OrdinaryDiffEq
using DiffEqDevTools
using SpecialMatrices, ToeplitzMatrices

# Current working solution to depending on functions that moved from DiffEqBase to SciMLBase
try
    DiffEqBase.interpret_vars
    DiffEqBase.getsyms # these are here to trigger an error
    import DiffEqBase: interpret_vars, getsyms
catch
    import DiffEqBase.SciMLBase: interpret_vars, getsyms
end
using SciMLBase

import Base: copy, copy!, show, size, ndims, similar
stack(x) = copy(reduce(hcat, x)')

using LinearAlgebra
import LinearAlgebra: mul!
"""LAPACK.geqrf! seems to be faster on small matrices than LAPACK.geqrt!"""
custom_qr!(A) = qr!(A)
# custom_qr!(A::StridedMatrix{<:LinearAlgebra.BlasFloat}) = QR(LAPACK.geqrf!(A)...)
using TaylorSeries
using TaylorIntegration
@reexport using StructArrays
using UnPack
using RecipesBase
using RecursiveArrayTools
using ForwardDiff
using ExponentialUtilities
using Octavian
# By default use mul!
_matmul!(C, A, B) = mul!(C, A, B)
_matmul!(C, A, B, a, b) = mul!(C, A, B, a, b)
# Some special cases
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::Diagonal{T},
) where {T<:LinearAlgebra.BlasFloat} = (C .= A .* B.diag')
_matmul!(
    C::AbstractMatrix{T},
    A::Diagonal{T},
    B::AbstractMatrix{T},
) where {T<:LinearAlgebra.BlasFloat} = (C .= A.diag .* B)
_matmul!(
    C::AbstractMatrix{T},
    A::Diagonal{T},
    B::Diagonal{T},
) where {T<:LinearAlgebra.BlasFloat} = @. C = A * B
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
    alpha::Number,
    beta::Number,
) where {T<:LinearAlgebra.BlasFloat} = matmul!(C, A, B, alpha, beta)
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
) where {T<:LinearAlgebra.BlasFloat} = matmul!(C, A, B)

@reexport using PSDMatrices
import PSDMatrices: X_A_Xt, X_A_Xt!
X_A_Xt(A, X) = X * A * X'
X_A_Xt!(out, A, X) = (out .= X * A * X')
apply_diffusion(Q, diffusion::Diagonal) = X_A_Xt(Q, sqrt.(diffusion))
apply_diffusion(Q::PSDMatrix, diffusion::Number) = PSDMatrix(sqrt.(diffusion) * Q.R)

# All the Gaussian things
@reexport using GaussianDistributions
using GaussianDistributions: logpdf

import Statistics: mean, var, std

include("gaussians.jl")

include("priors.jl")
export IWP, IOUP
include("diffusions.jl")
export FixedDiffusion, DynamicDiffusion, FixedMVDiffusion, DynamicMVDiffusion

include("initialization/common.jl")
export TaylorModeInit, ClassicSolverInit

include("algorithms.jl")
export EK0, EK1, EK1FDB

abstract type AbstractODEFilterCache <: OrdinaryDiffEq.OrdinaryDiffEqCache end
include("alg_utils.jl")
include("caches.jl")

include("checks.jl")

include("initialization/taylormode.jl")
include("initialization/classicsolverinit.jl")

include("solution.jl")
include("solution_sampling.jl")
include("solution_plotting.jl")

include("integrator_utils.jl")
include("filtering/predict.jl")
include("filtering/update.jl")
include("filtering/smooth.jl")
include("measurement_models.jl")
include("derivative_utils.jl")
include("perform_step.jl")
include("projection.jl")
include("solve.jl")

include("preconditioning.jl")

include("devtools.jl")
include("callbacks.jl")
export ManifoldUpdate

# Do as they do here:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/v6.21.0/src/OrdinaryDiffEq.jl#L195-L221
import SnoopPrecompile
SnoopPrecompile.@precompile_all_calls begin
    function lorenz(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
        return nothing
    end

    prob_list = [
        ODEProblem{true,true}(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0))
        ODEProblem{true,false}(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0))
        ODEProblem{true,false}(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0), Float64[])
    ]
    alg_list = [
        EK0()
        EK1()
    ]
    for prob in prob_list, solver in alg_list
        solve(prob, solver)(5.0)
    end
end

end
