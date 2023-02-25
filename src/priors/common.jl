abstract type AbstractODEFilterPrior{elType} end

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
initialize_transition_matrices(p::AbstractODEFilterPrior, dt)

function initialize_preconditioner(p::AbstractODEFilterPrior{T}, dt) where {T}
    d, q = p.wiener_process_dimension, p.num_derivatives
    P, PI = init_preconditioner(d, q, T)
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return P, PI
end
