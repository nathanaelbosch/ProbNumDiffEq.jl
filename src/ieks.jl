# Iterated extended Kalman smoothing
mutable struct IEKS <: AbstractEKF
    prior::Symbol
    order::Int
    diffusionmodel::Symbol
    smooth::Bool
    linearize_at
end

function IEKS(; prior=:ibm, order=1, diffusionmodel=:dynamic, linearize_at=nothing)
    if !isnothing(linearize_at)
        @assert linearize_at isa ProbODESolution
        @assert linearize_at.alg.prior == prior
        @assert linearize_at.alg.order == order
        @assert linearize_at.alg.diffusionmodel == diffusionmodel
        @assert linearize_at.alg.smooth == true
    end
    return IEKS(prior, order, diffusionmodel, true, linearize_at)
end


function solve_ieks(prob::DiffEqBase.AbstractODEProblem, alg::IEKS, args...;
                    iterations=10, kwargs...)
    sol = nothing
    for i in 1:iterations
        alg.linearize_at = sol
        sol = solve(prob, alg, args...; kwargs...)
    end
    return sol
end
