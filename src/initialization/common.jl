abstract type InitializationScheme end

"""
    TaylorModeInit()

Exact initialization via Taylor-mode automatic differentiation.

**This is the recommended initialization method!**

It uses [TaylorIntegration.jl](https://perezhz.github.io/TaylorIntegration.jl/latest/)
to efficiently compute the higher-order derivatives of the solution at the initial value,
via Taylor-mode automatic differentiation.

In some special cases it can happen that TaylorIntegration.jl is incompatible with the
given problem (typically because the problem definition does not allow for elements of type
 `Taylor`). If this happens, try [`ClassicSolverInit`](@ref).

# References
* [kraemer20stableimplementation](@cite) Krämer et al, "Stable Implementation of Probabilistic ODE Solvers" (2020)
"""
struct TaylorModeInit <: InitializationScheme end

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
    alg::ALG = Tsit5()
    init_on_ddu::Bool = false
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

    fast_X_A_Xt!(S, x.Σ, H)
    # @assert isdiag(Matrix(S))
    S_diag = diag(S)
    if any(iszero.(S_diag)) # could happen with a singular mass-matrix
        S_diag .+= 1e-20
    end

    _matmul!(K1, x.Σ.R', _matmul!(cache.C_Dxd, x.Σ.R, H'))
    K = K1 ./= S_diag'

    _K = x.Σ.R' * x.Σ.R * H'
    @assert all(S_diag .== 1)
    K = _K

    # x.μ .+= K*(data - z)
    datadiff = _matmul!(data, H, x.μ, -1, 1)
    _matmul!(x.μ, K, datadiff, 1, 1)

    D = length(x.μ)
    _matmul!(Mcache, K, H, -1, 0)
    @inbounds @simd ivdep for i in 1:D
        Mcache[i, i] += 1
    end

    d, q1 = size(H.A, 1), size(x.Σ.R.B, 1)
    _I = kronecker(I(d)*I(d), I(q1))
    KH = K*H
    @assert _I.A == KH.A
    @. KH.B = _I.B - KH.B
    M = KH

    fast_X_A_Xt!(x_tmp.Σ, x.Σ, M)
    copy!(x.Σ.R.A, covcache.R.A)
    copy!(x.Σ.R.B, covcache.R.B)
    return nothing
end
