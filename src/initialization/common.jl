abstract type InitializationScheme end
abstract type AutodiffInitializationScheme <: InitializationScheme end

"""
    SimpleInit()

Simple initialization, only with the given initial value and derivative.

The remaining derivatives are set to zero with unit covariance (unless specified otherwise
by setting a custom [`FixedDiffusion`](@ref)).
"""
struct SimpleInit <: InitializationScheme end

"""
    TaylorModeInit(order)

Exact initialization via Taylor-mode automatic differentiation up to order `order`.

The `order` is a required argument and must be at least 1; usually it is set to the order
of the prior, e.g. `EK1(order=3, initialization=TaylorModeInit(3))`. Calling
`TaylorModeInit()` without an order throws an `ArgumentError`.

**This is the recommended initialization method!**

It uses [TaylorIntegration.jl](https://perezhz.github.io/TaylorIntegration.jl/latest/)
to efficiently compute the higher-order derivatives of the solution at the initial value,
via Taylor-mode automatic differentiation.

In some special cases it can happen that TaylorIntegration.jl is incompatible with the
given problem (typically because the problem definition does not allow for elements of type
 `Taylor`). If this happens, try one of [`SimpleInit`](@ref), [`ForwardDiffInit`](@ref)
(for low enough orders), [`ClassicSolverInit`](@ref).

# References
* [Krämer & Hennig (2020)](@cite kraemer20stableimplementation) "Stable Implementation of Probabilistic ODE Solvers"
"""
struct TaylorModeInit <: AutodiffInitializationScheme
    order::Int64
    TaylorModeInit(order::Int64) = begin
        if order < 1
            throw(ArgumentError("order must be >= 1"))
        end
        new(order)
    end
end
TaylorModeInit() = throw(
    ArgumentError(
        "`TaylorModeInit` requires an `order` argument, e.g. `TaylorModeInit(3)`",
    ),
)

"""
    ForwardDiffInit(order)

Exact initialization via ForwardDiff.jl up to order `order`.

The `order` is a required argument and must be at least 1; usually it is set to the order
of the prior, e.g. `EK1(order=3, initialization=ForwardDiffInit(3))`. Calling
`ForwardDiffInit()` without an order throws an `ArgumentError`.

**Warning:** This does not scale well to high orders!
For orders > 3, [`TaylorModeInit`](@ref) most likely performs better.
"""
struct ForwardDiffInit <: AutodiffInitializationScheme
    order::Int64
    ForwardDiffInit(order::Int64) = begin
        if order < 1
            throw(ArgumentError("order must be >= 1"))
        end
        new(order)
    end
end
ForwardDiffInit() = throw(
    ArgumentError(
        "`ForwardDiffInit` requires an `order` argument, e.g. `ForwardDiffInit(3)`",
    ),
)

"""
    ClassicSolverInit(; alg, init_on_ddu=false)

Initialization via regression on a few steps of a classic ODE solver.

In a nutshell, instead of specifying ``\\mu_0`` exactly and setting ``\\Sigma_0=0`` (which
is what [`TaylorModeInit`](@ref) does), use a classic ODE solver to compute a few steps
of the solution, and then regress on the computed values (by running a smoother) to compute
``\\mu_0`` and ``\\Sigma_0`` as the mean and covariance of the smoothing posterior at
time 0. See also [[2]](@ref initrefs).

The initial value and derivative are set directly from the given initial value problem;
optionally the second derivative can also be set via automatic differentiation by setting
`init_on_ddu=true`.

# Arguments
- `alg`: The solver to be used. Can be any solver from OrdinaryDiffEq.jl (or one of its
  sub-packages). If you don't know whether your problem is stiff, a robust choice is
  `AutoTsit5(Rosenbrock23())`, which automatically switches between the two depending on
  the detected stiffness; this is also what plain `solve(prob)` uses internally when no
  algorithm is specified.
- `init_on_ddu`: If `true`, the second derivative is also initialized exactly via
  automatic differentiation with ForwardDiff.jl.

# References
* [Krämer & Hennig (2020)](@cite kraemer20stableimplementation) "Stable Implementation of Probabilistic ODE Solvers"
* [Schober et al. (2019)](@cite schober16probivp) "A probabilistic model for the numerical solution of initial value problems", Statistics and Computing
"""
Base.@kwdef struct ClassicSolverInit{ALG} <: InitializationScheme
    alg::ALG
    init_on_ddu::Bool = false
end
ClassicSolverInit(alg::SciMLBase.AbstractODEAlgorithm) = ClassicSolverInit(; alg)

"""
    _unwrap_f(f)

Strip the `FunctionWrappersWrapper` that OrdinaryDiffEq puts around `f.f`, so that `f` can
be called with arguments of other element types (e.g. `ForwardDiff.Dual` or `Taylor1`).

If `f` is an `ODEFunction` whose `f.f` is a `FunctionWrappersWrapper`, returns
`SciMLBase.unwrapped_f(f)`: the same `ODEFunction` with only `f.f` unwrapped, and all other
fields (e.g. `jac`, `mass_matrix`) unchanged. Otherwise returns `f` unchanged.
"""
function _unwrap_f(f)
    if f isa ODEFunction &&
       f.f isa SciMLBase.FunctionWrappersWrappers.FunctionWrappersWrapper
        return SciMLBase.unwrapped_f(f)
    end
    return f
end

"""
    initial_update!(integ, cache[, init::InitializationScheme])

Improve the initial state estimate by updating either on exact derivatives or values
computed with a classic solver.

See also: [Initialization](@ref), [`TaylorModeInit`](@ref), [`ClassicSolverInit`](@ref).
"""
function initial_update!(integ, cache)
    return initial_update!(integ, cache, integ.alg.initialization)
end

"""
    init_condition_on!(x, H, data, cache)

Condition `x` on `data` with linear measurement function `H`. Used only for initialization.

Rows of `H` that are zero, such as those of `M * Proj(o)` for a mass matrix `M` with zero
rows, are treated as unobserved; see [`_zero_row_noise`](@ref).

Don't use this as a Kalman update! The function has quite a few assumptions, that only
really work out in the specific context of initialization. If you actually want to update,
use [`update`](@ref) or [`update!`](@ref).
"""
function init_condition_on!(
    x::SRGaussian,
    H::AbstractMatrix,
    data::AbstractVector,
    cache,
)
    @unpack x_tmp, m_tmp = cache
    z = _matmul!(m_tmp.μ, H, x.μ)
    z .-= data
    copy!(x_tmp, x)
    obs = LinearizedObservation(x_tmp.μ, z, H, _zero_row_noise(H, cache))
    return update!(x, x_tmp, obs; cache)
end

"""
    _zero_row_noise(H, cache)

Unit observation noise on the zero rows of `H`, or `nothing` if `H` has no zero row.

A zero row `i` of `H` makes row and column `i` of `S = H Σ Hᵀ` zero, so `S` is singular.
With unit noise on row `i`, `S[i, i] = 1`, `S` can be factorized, and column `i` of the gain
`Σ Hᵀ S⁻¹` is zero, so the observation in this row is ignored.
"""
function _zero_row_noise(H, cache)
    R = _ones_on_zero_rows!(zero(cache.C_dxd), H)
    return iszero(R) ? nothing : PSDMatrix(R)
end
function _ones_on_zero_rows!(R::AbstractMatrix, H::AbstractMatrix)
    for i in axes(H, 1)
        if iszero(view(H, i, :))
            R[i, i] = 1
        end
    end
    return R
end
_ones_on_zero_rows!(R::IsometricKroneckerProduct, H::IsometricKroneckerProduct) =
    (on_kronecker_factors(_ones_on_zero_rows!, H.rdim, R, H); R)
_ones_on_zero_rows!(R::BlocksOfDiagonals, H::BlocksOfDiagonals) =
    (foreach_diagonal_block(_ones_on_zero_rows!, nblocks(H), R, H); R)
