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
    update_on_derivative!(x, M, i, y; cache)

Condition the initial state `x` on `M u⁽ⁱ⁾ = y`, for the `i`-th derivative `u⁽ⁱ⁾` of the
solution and the mass matrix `M`. The zero rows of `M`, the algebraic equations of a DAE, say
nothing about `u⁽ⁱ⁾` and would make `S = H Σ Hᵀ` singular, so the observation leaves them
out.
"""
function update_on_derivative!(x, M, i, y; cache)
    H, rows = _derivative_observation(cache.covariance_factorization, cache, M, i)
    o = length(rows)
    o == 0 && return nothing
    y = o == cache.d ? y : view(y, rows)
    return update_on_data!(x, H, y, nothing; cache)
end

# `H` for the nonzero `rows` of `M`
function _derivative_observation(::DenseCovariance, cache, M, i)
    H = M * cache.Proj(i)
    rows = findall(!iszero, eachrow(H))
    return length(rows) == cache.d ? H : H[rows, :], rows
end
_derivative_observation(::IsometricKroneckerCovariance, cache, M::UniformScaling, i) =
    M * cache.Proj(i), iszero(M.λ) ? (1:0) : (1:cache.d)
function _derivative_observation(C::BlockDiagonalCovariance, cache, M, i)
    m = M isa UniformScaling ? fill(M.λ, cache.d) : M.diag
    rows = findall(!iszero, m)
    length(rows) == cache.d && return M * cache.Proj(i), rows
    H = SolutionObservation(ScaledSelection(m[rows], rows, cache.d), cache.q, i)
    return to_factorized_matrix(C, H), rows
end
