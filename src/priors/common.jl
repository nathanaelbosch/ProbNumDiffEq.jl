abstract type AbstractODEFilterPrior{elType} end

function initialize_preconditioner(p::AbstractODEFilterPrior{T}, dt) where {T}
    d, q = p.wiener_process_dimension, p.num_derivatives
    P, PI = init_preconditioner(d, q, T)
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return P, PI
end

"""
    initilize_transition_matrices!(p::AbstractODEFilterPrior)

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
function initialize_transition_matrices(p::AbstractODEFilterPrior{T}, dt) where {T}
    Ah, Qh = discretize(p, dt)
    P, PI = initialize_preconditioner(p, dt)
    A = P * Ah * PI
    Q = X_A_Xt(Qh, P)
    return A, Q, Ah, Qh, P, PI
end

"""
    make_transition_matrices!(cache, prior::AbstractODEFilterPrior, dt)

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
function make_transition_matrices!(cache, prior::AbstractODEFilterPrior, dt)
    @unpack A, Q, Ah, Qh, P, PI = cache
    make_preconditioners!(cache, dt)
    _Ah, _Qh = discretize(cache.prior, dt)
    copy!(Ah, _Ah)
    copy!(Qh, _Qh)
    A .= P.diag .* Ah .* PI.diag'
    fast_X_A_Xt!(Q, Qh, P)
end

# convenience function
make_transition_matrices!(cache::AbstractODEFilterCache, dt) =
    make_transition_matrices!(cache, cache.prior, dt)

"""
    to_1d_sde(p::AbstractODEFilterPrior)

Convert the prior to a 1-dimensional SDE. This is only possible for independent dimensions.
"""
to_1d_sde(p::AbstractODEFilterPrior)

function to_sde(p::AbstractODEFilterPrior)
    d = p.wiener_process_dimension
    _sde = to_1d_sde(p)
    F_breve, L_breve = drift(_sde), dispersion(_sde)
    F = kron(I(d), F_breve)
    L = kron(I(d), L_breve)
    return LTISDE(F, L)
end
