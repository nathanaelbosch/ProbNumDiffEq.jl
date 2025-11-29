@doc raw"""
    AbstractGaussMarkovProcess{elType}

Abstract type for Gauss-Markov processes.

Gauss-Markov processes are solutions to linear time-invariant stochastic differential
equations (SDEs). Here we assume SDEs of the form
```math
\begin{aligned}
dX_t &= F X_t dt + L dW_t \\
X_0 &= \mathcal{N} \left( X_0; \mu_0, \Sigma_0 \right)
\end{aligned}
```
where ``X_t`` is the state, ``W_t`` is a Wiener process, and ``F`` and ``L`` are matrices.

Currently, ProbNumDiffEq.jl makes many assumptions about the structure of the SDEs that it
can solve. In particular, it assumes that the state vector ``X_t`` contains a range of
dervatives, and that the Wiener process only enters the highest one. It also assumes a
certain ordering of dimensions and derivatives. This is not a limitation of the underlying
mathematics, but rather a limitation of the current implementation. In the future, we hope
to remove these limitations.

!!! warning
    We currently strongly recommended to not implement your own Gauss-Markov process by
    subtyping this type! The interface is not yet stable, and the implementation is not yet
    sufficiently documented. Proceed at your own risk.
"""
abstract type AbstractGaussMarkovProcess{elType} end

############################################################################################
# Interface
############################################################################################
"""
    dim(p::AbstractGaussMarkovProcess)

Return the dimension of the process.

This is not the dimension of the "state" that is used to efficiently model the prior
process as a state-space model, but it is the dimension of the process itself that we aim
to model.

See [`AbstractGaussMarkovProcess`](@ref) for more details on Gauss-Markov processes in ProbNumDiffEq.
"""
dim(p::AbstractGaussMarkovProcess) = p.dim

"""
    num_derivatives(p::AbstractGaussMarkovProcess)

Return the number of derivatives that are represented by the processes state.

See [`AbstractGaussMarkovProcess`](@ref) for more details on Gauss-Markov processes in ProbNumDiffEq.
"""
num_derivatives(p::AbstractGaussMarkovProcess) = p.num_derivatives

"""
    discretize(p::AbstractGaussMarkovProcess, step_size::Real)

Compute the transition matrices of the process for a given step size.
"""
discretize(p::AbstractGaussMarkovProcess, step_size::Real) =
    discretize(to_sde(p), step_size)

"""
    initial_distribution(p::AbstractGaussMarkovProcess)

Return the initial distribution of the process.

Currently this is always a Gaussian distribution with zero mean and unit variance, unless
explicitly overwitten (e.g. for Matern processes to have the stationary distribution).
This implementation is likely to change in the future to allow for more flexibility.
"""
initial_distribution(p::AbstractGaussMarkovProcess{T}) where {T} = begin
    d, q = dim(p), num_derivatives(p)
    D = d * (q + 1)
    initial_variance = ones(T, q + 1)
    μ0 = T <: LinearAlgebra.BlasFloat ? Array{T}(calloc, D) : zeros(T, D)
    Σ0 = PSDMatrix(IsometricKroneckerProduct(d, diagm(sqrt.(initial_variance))))
    return Gaussian(μ0, Σ0)
end

"""
    SciMLBase.remake(::AbstractGaussMarkovProcess{T}; eltype=T, kwargs...)

Create a new process of the same type, but with different parameters.
This is particularly used to set the Wiener process dimension, so that the prior can be
defined with missing dimension first, and then have the dimension set to the dimension of
the ODE. This corresponds to having the same prior for all dimensions of the ODE.
Similarly, the element type of the process is also set to the element type of the ODE.
"""
remake(p::AbstractGaussMarkovProcess{T}; elType=T, kwargs...) where {T}

@doc raw"""
    to_sde(p::AbstractGaussMarkovProcess)

Convert the prior to the corresponding SDE.

Gauss-Markov processes are solutions to linear time-invariant stochastic differential
equations (SDEs) of the form
```math
\begin{aligned}
dX_t &= F X_t dt + L dW_t \\
X_0 &= \mathcal{N} \left( X_0; \mu_0, \Sigma_0 \right)
\end{aligned}
```
where ``X_t`` is the state, ``W_t`` is a Wiener process, and ``F`` and ``L`` are matrices.
This function returns the corresponding SDE, i.e. the matrices ``F`` and ``L``, as a
[`LTISDE`](@ref).
"""
to_sde(p::AbstractGaussMarkovProcess)

############################################################################################
# General implementations
############################################################################################
function initialize_preconditioner(
    FAC::CovarianceStructure{T1}, p::AbstractGaussMarkovProcess{T}, dt) where {T,T1}
    @assert T == T1
    d, q = dim(p), num_derivatives(p)
    P, PI = init_preconditioner(FAC)
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return P, PI
end

"""
    initilize_transition_matrices!(p::AbstractGaussMarkovProcess)

Create all the (moslty empty) matrices that relate to the transition model.

The transition model (specified in `cache.prior`) is of the form
```math
X(t+h) \\mid X(t) \\sim \\mathcal{N} \\left( X(t+h); A(h) X(t), Q(h) \\right).
```
In addition, for improved numerical stability it computes preconditioning matrices ``P, P^{-1}`` as described in [1], as well as transition matrices
```math
\\begin{aligned}
A = P A(h) P^{-1}, \\\\
Q = P Q(h) P.\\\\
\\end{aligned}
```
This function creates matrices `A`, `Q`, `Ah`, `Qh`, `P`, `PI` to hold all these quantities, and returns them.
The matrices will then be filled with the correct values later, at each solver step, by calling [`make_transition_matrices`](@ref).

See also: [`make_transition_matrices`](@ref).

[1] N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020)
"""
function initialize_transition_matrices(
    FAC::DenseCovariance,
    p::AbstractGaussMarkovProcess{T},
    dt,
) where {T}
    d, q = dim(p), num_derivatives(p)
    D = d * (q + 1)
    Ah, Qh = zeros(T, D, D), PSDMatrix(zeros(T, D, D))
    P, PI = initialize_preconditioner(FAC, p, dt)
    A = copy(Ah)
    Q = copy(Qh)
    return A, Q, Ah, Qh, P, PI
end
initialize_transition_matrices(
    FAC::CovarianceStructure,
    p::AbstractGaussMarkovProcess,
    dt,
) =
    error("The chosen prior can not be implemented with a $FAC factorization")

"""
    make_transition_matrices!(cache, prior::AbstractGaussMarkovProcess, dt)

Construct all the matrices that relate to the transition model, for a specified step size.

The transition model (specified in `cache.prior`) is of the form
```math
X(t+h) \\mid X(t) \\sim \\mathcal{N} \\left( X(t+h); A(h) X(t), Q(h) \\right).
```
This function constructs ``A(h)`` and ``Q(h)`` and writes them into `cache.Ah` and `cache.Qh`.

In addition, for improved numerical stability it computes preconditioning matrices ``P, P^{-1}`` as described in [1], as well as transition matrices
```math
\\begin{aligned}
A = P A(h) P^{-1}, \\\\
Q = P Q(h) P.\\\\
\\end{aligned}
```
The preconditioning matrices and the preconditioned state transition matrices are saved in `cache.P, cache.PI, cache.A, cache.Q`.`

Note that `cache` would typically be an `EKCache`, but the function also works for any type that has fields `Ah, Qh, A, Q, P, PI`.

See also: [`initialize_transition_matrices`](@ref).

[1] N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020)
"""
function make_transition_matrices!(cache, prior::AbstractGaussMarkovProcess, dt)
    @unpack A, Q, Ah, Qh, P, PI = cache
    make_preconditioners!(cache, dt)
    _Ah, _Qh = discretize(prior, dt)
    copy!(Ah, _Ah)
    copy!(Qh, _Qh)
    # A = P * Ah * PI
    _matmul!(A, P, _matmul!(A, Ah, PI))
    fast_X_A_Xt!(Q, Qh, P)
end

# convenience function
make_transition_matrices!(cache::AbstractODEFilterCache, dt) =
    make_transition_matrices!(cache, cache.prior, dt)

"""
    marginalize(process::AbstractGaussMarkovProcess, times)

Compute the marginal distributions of the process at the given time points.

This function computes the marginal distributions of the process at the given times.
It does so by discretizing the process with the given step sizes (using
[`ProbNumDiffEq.discretize`](@ref)), and then computing the marginal distributions of the
resulting Gaussian distributions.

See also: [`sample`](@ref).
"""
function marginalize(process::AbstractGaussMarkovProcess, times)
    X = initial_distribution(process)
    out = Gaussian[X]
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        A, Q = ProbNumDiffEq.discretize(process, dt)
        X = predict(X, A, Q)
        push!(out, X)
    end
    return out
end

"""
    sample(process::AbstractGaussMarkovProcess, times, N=1)

Samples from the Gauss-Markov process on the given time grid.

See also: [`marginalize`](@ref).
"""
function sample(process::AbstractGaussMarkovProcess, times, N::Integer=1)
    X = initial_distribution(process)
    X = Gaussian(mean(X), Matrix(cov(X)))
    s = rand(X, N)
    out = [s]
    for i in 2:length(times)
        dt = times[i] - times[i-1]
        A, Q = Matrix.(discretize(process, dt))
        s = [rand(Gaussian(A * s[:, j], Q)) for j in 1:N] |> stack |> permutedims
        push!(out, s)
    end
    return out
end
