abstract type InitializationScheme end

"""
    TaylorModeInit()

**Recommended**

Exact initialization via Taylor-mode automatic differentiation.
Uses [TaylorIntegration.jl](https://perezhz.github.io/TaylorIntegration.jl/latest/).
In case of errors, try [`ClassicSolverInit`](@ref).

# References:
- N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020)
"""
struct TaylorModeInit <: InitializationScheme end

"""
    ClassicSolverInit(; alg=OrdinaryDiffEq.Tsit5(), init_on_du=false)

Exact initialization with a classic ODE solver. The solver to be used can be set with the
`alg` keyword argument. `init_on_du` specifies if ForwardDiff.jl should be used to compute
the jacobian and initialize on the exact second derivative.

Not recommended for large solver orders, say `order>4`.
"""
Base.@kwdef struct ClassicSolverInit{ALG} <: InitializationScheme
    alg::ALG = Tsit5()
    init_on_du::Bool = false
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
    condition_on!(x, H, data, cache)

Condition `x` on `data`, with linearized measurement function `H`.

This is basically a Kalman update. We recommend using [`update`](@ref) or [`update!`](@ref).
"""
function condition_on!(
    x::SRGaussian,
    H::AbstractMatrix,
    data::AbstractVector,
    cache,
)
    @unpack m_tmp, K1, x_tmp, C_DxD = cache
    S = m_tmp.Σ
    covcache = x_tmp.Σ
    Mcache = cache.C_DxD

    X_A_Xt!(S, x.Σ, H)
    # @assert isdiag(Matrix(S))
    S_diag = diag(S)
    if any(iszero.(S_diag)) # could happen with a singular mass-matrix
        S_diag .+= 1e-20
    end

    _matmul!(K1, x.Σ.R', _matmul!(cache.C_Dxd, x.Σ.R, H'))
    K = K1 ./= S_diag'

    # x.μ .+= K*(data - z)
    datadiff = _matmul!(data, H, x.μ, -1, 1)
    _matmul!(x.μ, K, datadiff, 1, 1)

    D = length(x.μ)
    _matmul!(Mcache, K, H, -1, 0)
    @inbounds @simd ivdep for i in 1:D
        Mcache[i, i] += 1
    end
    X_A_Xt!(x_tmp.Σ, x.Σ, Mcache)
    copy!(x.Σ, covcache)
    return nothing
end
