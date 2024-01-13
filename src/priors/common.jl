abstract type AbstractGaussMarkovProcess{elType} end

# Fields they should have and Interface
wiener_process_dimension(p::AbstractGaussMarkovProcess) = p.wiener_process_dimension
num_derivatives(p::AbstractGaussMarkovProcess) = p.num_derivatives
to_sde(p::AbstractGaussMarkovProcess) = missing
discretize(p::AbstractGaussMarkovProcess, step_size::Real) =
    discretize(to_sde(p), step_size)
initial_distribution(p::AbstractGaussMarkovProcess{T}) where {T} = begin
    d, q = wiener_process_dimension(p), num_derivatives(p)
    D = d * (q + 1)
    initial_variance = ones(T, q + 1)
    μ0 = T <: LinearAlgebra.BlasFloat ? Array{T}(calloc, D) : zeros(T, D)
    Σ0 = PSDMatrix(IsometricKroneckerProduct(d, diagm(sqrt.(initial_variance))))
    return Gaussian(μ0, Σ0)
end

""
remake(p::AbstractGaussMarkovProcess{T}; elType=T, kwargs...) where {T}

"""
    to_sde(p::AbstractGaussMarkovProcess)

Convert the prior to the corresponding SDE.
"""
to_sde(p::AbstractGaussMarkovProcess)

# General implementations
function initialize_preconditioner(
    FAC::CovarianceStructure{T1}, p::AbstractGaussMarkovProcess{T}, dt) where {T,T1}
    @assert T == T1
    d, q = wiener_process_dimension(p), num_derivatives(p)
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
    d, q = wiener_process_dimension(p), num_derivatives(p)
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
