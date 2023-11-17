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

**This is the recommended initialization method!**

It uses [TaylorIntegration.jl](https://perezhz.github.io/TaylorIntegration.jl/latest/)
to efficiently compute the higher-order derivatives of the solution at the initial value,
via Taylor-mode automatic differentiation.

In some special cases it can happen that TaylorIntegration.jl is incompatible with the
given problem (typically because the problem definition does not allow for elements of type
 `Taylor`). If this happens, try one of [`SimpleInit`](@ref), [`ForwardDiffInit`](@ref)
(for low enough orders), [`ClassicSolverInit`](@ref).

# References
* [kraemer20stableimplementation](@cite) Krämer et al, "Stable Implementation of Probabilistic ODE Solvers" (2020)
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
TaylorModeInit() = begin
    throw(ArgumentError("order must be specified"))
end

"""
    ForwardDiffInit(order)

Exact initialization via ForwardDiff.jl up to order `order`.

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
ForwardDiffInit() = begin
    throw(ArgumentError("order must be specified"))
end

"""
    ClassicSolverInit(; alg=OrdinaryDiffEq.Tsit5(), init_on_ddu=false)

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
- `alg`: The solver to be used. Can be any solver from OrdinaryDiffEq.jl.
- `init_on_ddu`: If `true`, the second derivative is also initialized exactly via
  automatic differentiation with ForwardDiff.jl.

# References
* [kraemer20stableimplementation](@cite) Krämer et al, "Stable Implementation of Probabilistic ODE Solvers" (2020)
* [schober16probivp](@cite) Schober et al, "A probabilistic model for the numerical solution of initial value problems", Statistics and Computing (2019)
"""
Base.@kwdef struct ClassicSolverInit{ALG} <: InitializationScheme
    alg::ALG = AutoVern7(Rodas4())
    init_on_ddu::Bool = false
end
ClassicSolverInit(alg::DiffEqBase.AbstractODEAlgorithm) = ClassicSolverInit(; alg)

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
    @unpack x_tmp, K1, C_Dxd, C_DxD, C_dxd, m_tmp = cache

    # measurement mean
    _matmul!(m_tmp.μ, H, x.μ)
    m_tmp.μ .-= data

    # measurement cov
    fast_X_A_Xt!(m_tmp.Σ, x.Σ, H)
    copy!(x_tmp, x)
    update!(x, x_tmp, m_tmp, H, K1, C_Dxd, C_DxD, C_dxd)
end
