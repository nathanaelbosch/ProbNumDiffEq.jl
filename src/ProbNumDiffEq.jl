__precompile__()

module ProbNumDiffEq

# All imports are explicit and from the owner module (not from re-exporting middlemen),
# so that upstream `using`-to-explicit-import cleanups cannot silently remove our access
# paths. Enforced by the ExplicitImports checks in the CodeQuality test group.
import Base:
    copy, copy!, show, size, similar, isapprox, isequal, iterate, ==, length, zero,
    eltype, rand

using LinearAlgebra: LinearAlgebra, Adjoint, Cholesky, Diagonal, I, QR, Symmetric,
    UniformScaling, UpperTriangular, cholesky, cholesky!, diag, diagm, dot, ishermitian,
    issuccess, ldiv!, logdet, norm, qr, qr!, rdiv!, rmul!, triu!
import LinearAlgebra: mul!
import Statistics: mean, var, std, cov
import Random: Random, AbstractRNG
using Printf: Printf, @printf
using DocStringExtensions: DocStringExtensions, TYPEDEF, TYPEDSIGNATURES

using Reexport: Reexport, @reexport
@reexport using DiffEqBase
using DiffEqBase: DiffEqBase
import SciMLBase
import SciMLBase: remake
using SciMLBase: DAEFunction, DiscreteCallback, DynamicalODEFunction, ODEFunction,
    ODEProblem, ReturnCode, SecondOrderODEProblem, isinplace
using CommonSolve: init, solve, step!
import SciMLOperators
using SciMLOperators: MatrixOperator
const WOperator = SciMLOperators.WOperator # to fix an Aqua.jl undefined export test
import ConstructionBase
using OrdinaryDiffEqCore: OrdinaryDiffEqCore
using OrdinaryDiffEqDifferentiation: OrdinaryDiffEqDifferentiation
using OrdinaryDiffEqVerner: OrdinaryDiffEqVerner, AutoVern7
using OrdinaryDiffEqRosenbrock: OrdinaryDiffEqRosenbrock, Rodas4
using ToeplitzMatrices: ToeplitzMatrices, TriangularToeplitz
using FastBroadcast: FastBroadcast, @..
using StaticArrayInterface: StaticArrayInterface
using FunctionWrappersWrappers: FunctionWrappersWrappers
using TaylorSeries: TaylorSeries, Taylor1, differentiate, evaluate, order
using TaylorIntegration: TaylorIntegration
@reexport using StructArrays
using StructArrays: StructArrays, StructArray
using SimpleUnPack: SimpleUnPack, @unpack
using RecursiveArrayTools: RecursiveArrayTools, ArrayPartition, DiffEqArray,
    copyat_or_push!, recursive_unitless_bottom_eltype, recursive_unitless_eltype,
    recursivecopy, recursivecopy!
using ForwardDiff: ForwardDiff
using DiffResults: DiffResults
using Octavian: Octavian, matmul!
import Kronecker
using ArrayAllocators: ArrayAllocators, calloc
using FiniteHorizonGramians: FiniteHorizonGramians
using FillArrays: FillArrays, Eye, Fill
using MatrixEquations: MatrixEquations, plyapc
using DiffEqCallbacks: DiffEqCallbacks, PresetTimeCallback
using ADTypes: ADTypes, AutoForwardDiff, AutoSparse

# @reexport using GaussianDistributions

@reexport using PSDMatrices
using PSDMatrices: PSDMatrices, PSDMatrix
import PSDMatrices: X_A_Xt, X_A_Xt!

stack(x) = copy(reduce(hcat, x)')
vecvec2mat(x) = reduce(hcat, x)'

cov2psdmatrix(cov::Number; d) = PSDMatrix(sqrt(cov) * Eye(d))
cov2psdmatrix(cov::UniformScaling; d) = PSDMatrix(sqrt(cov.λ) * Eye(d))
cov2psdmatrix(cov::Diagonal{<:Number,<:FillArrays.Fill}; d) =
    (@assert size(cov, 1) == size(cov, 2) == d; cov2psdmatrix(cov.diag.value; d))
cov2psdmatrix(cov::Diagonal; d) =
    (@assert size(cov, 1) == size(cov, 2) == d; PSDMatrix(sqrt.(cov)))
cov2psdmatrix(cov::AbstractMatrix; d) =
    (@assert size(cov, 1) == size(cov, 2) == d; PSDMatrix(Matrix(cholesky(cov).U)))
cov2psdmatrix(cov::PSDMatrix; d) = (@assert size(cov, 1) == size(cov, 2) == d; cov)
make_hermitian_if_fowarddiff(M::AbstractMatrix) = M
make_hermitian_if_fowarddiff(M::AbstractMatrix{<:ForwardDiff.Dual}) = begin
    # Since ForwardDiff 1.0 `ishermitian` fails for small fluctuations in the dual
    if !ishermitian(ForwardDiff.value.(M))
        error("Matrix is not Hermitian.")
    end
    return 0.5 * (M + M')
end

"""
    add!(out, toadd)

Add `toadd` to `out` in-place.
"""
add!
add!(out, toadd) = (out .+= toadd)

include("fast_linalg.jl")
include("kronecker.jl")
include("blocksofdiagonals.jl")
include("covariance_structure.jl")
export IsometricKroneckerCovariance, DenseCovariance, BlockDiagonalCovariance

abstract type AbstractODEFilterCache <: OrdinaryDiffEqCore.OrdinaryDiffEqCache end

include("gaussians.jl")
export Gaussian

include("priors/common.jl")
include("priors/iwp.jl")
include("priors/ltisde.jl")
include("priors/ioup.jl")
include("priors/matern.jl")
export IWP, IOUP, Matern
include("diffusions/typedefs.jl")
include("diffusions/apply_diffusion.jl")
include("diffusions/calibration.jl")
export FixedDiffusion, DynamicDiffusion, FixedMVDiffusion, DynamicMVDiffusion

include("initialization/common.jl")
export TaylorModeInit, ClassicSolverInit, SimpleInit, ForwardDiffInit

include("algorithms.jl")
export EK0, EK1, DiagonalEK1
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

include("callbacks/manifoldupdate.jl")
export ManifoldUpdate
include("callbacks/dataupdate.jl")
export DataUpdateLogLikelihood, DataUpdateCallback

include("data_likelihoods/dalton.jl")
include("data_likelihoods/filtering.jl")
include("data_likelihoods/fenrir.jl")
module DataLikelihoods
import ..ProbNumDiffEq: dalton_data_loglik, filtering_data_loglik, fenrir_data_loglik
export dalton_data_loglik, filtering_data_loglik, fenrir_data_loglik
end

include("precompile.jl")

end
