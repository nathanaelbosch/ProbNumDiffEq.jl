# Iterated extended Kalman smoothing
mutable struct IEKS{DT,IT} <: AbstractEK
    prior::Symbol
    order::Int
    diffusionmodel::DT
    smooth::Bool
    initialization::IT
    linearize_at::Any
end

"""
    IEKS(; prior=:ibm, order=1, diffusionmodel=DynamicDiffusion(), initialization=TaylorModeInit(), linearize_at=nothing)

**Gaussian ODE filtering with iterated extended Kalman smoothing.**

To use it, use
`solve_ieks(prob, IEKS(), args...)`
instead of
`solve(prob, IEKS(), args...)`,
since it is implemented as an outer loop around the solver.

Currently, only the integrated Brownian motion prior `:ibm` is supported.
For the diffusionmodel, chose one of
`[DynamicDiffusion(), DynamicMVDiffusion(), FixedDiffusion(), FixedMVDiffusion()]`.
Just like the [`EK1`](@ref) it requires that the Jacobian of the rhs function is available.

See also: [`EK0`](@ref), [`EK1`](@ref), [`solve_ieks`](@ref)

# References:
- F. Tronarp, S. Särkkä, and P. Hennig: **Bayesian ODE Solvers: The Maximum A Posteriori Estimate**
"""
function IEKS(;
    prior=:ibm,
    order=1,
    diffusionmodel=DynamicDiffusion(),
    initialization=TaylorModeInit(),
    linearize_at=nothing,
)
    if !isnothing(linearize_at)
        @assert linearize_at isa ProbODESolution
        @assert linearize_at.alg.prior == prior
        @assert linearize_at.alg.order == order
        @assert linearize_at.alg.diffusionmodel == diffusionmodel
        @assert linearize_at.alg.initialization == initialization
        @assert linearize_at.alg.smooth == true
    end
    return IEKS(prior, order, diffusionmodel, true, initialization, linearize_at)
end

"""
    solve_ieks(prob::AbstractODEProblem, alg::IEKS, args...; iterations=10, kwargs...)

Solve method to be used with the [`IEKS`](@ref). The IEKS works essentially by solving the
ODE multiple times. `solve_ieks` therefore wraps a call to the standard `solve` method,
passing `args...` and `kwargs...`.

Currently, this method is very simplistic - it iterates for a fixed numer of times and does
not use a stopping criterion.
"""
function solve_ieks(
    prob::DiffEqBase.AbstractODEProblem,
    alg::IEKS,
    args...;
    iterations=10,
    kwargs...,
)
    sol = nothing
    for i in 1:iterations
        alg.linearize_at = sol
        sol = solve(prob, alg, args...; kwargs...)
    end
    return sol
end
