########################################################################################
# Structure following https://github.com/SciML/SimpleDiffEq.jl/tree/master/src/rk4
########################################################################################
import DiffEqBase: __solve

abstract type AbstractODEFilter <: DiffEqBase.AbstractODEAlgorithm end
mutable struct ODEFilter <: AbstractODEFilter
end
export ODEFilter


function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::ODEFilter;
                            dt=nothing,
                            sigmarule=Schober16Sigma(),
                            steprule=:baseline,
                            kwargs...)
    if isnothing(dt) && steprule!=:constant
        dt = prob.tspan[2] - prob.tspan[1]
    elseif isnothing(dt)
        dt = 0.01
    end

    prob_sol = prob_solve(prob, dt; sigmarule=sigmarule, steprule=steprule, kwargs...)

    d = prob_sol.solver.d
    ts = prob_sol.t
    function make_Measurement(u)
        @assert isdiag(u.Σ[1:d,1:d]) u.Σ[1:d,1:d]
        return u.μ[1:d] .± sqrt.(diag(u.Σ)[1:d])
    end
    us = map(make_Measurement, prob_sol.u)
    sol = DiffEqBase.build_solution(prob, alg, ts, prob_sol.u)

    return sol
end


########################################################################################
# Solution handling
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,xType,tType,P,A,IType} <: AbstractProbODESolution{T,N,uType}
    u::uType
    x::xType
    t::tType
    prob::P
    alg::A
    dense::Bool
    interp::IType
    retcode::Symbol
end

function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ODEFilter,
    t,x;
    dense=false,
    retcode = :Default,
    kwargs...)

    d = length(prob.u0)
    function make_Measurement(state)
        @assert isdiag(state.Σ[1:d,1:d]) state.Σ[1:d,1:d]
        return state.μ[1:d] .± sqrt.(diag(state.Σ)[1:d])
    end
    u = map(make_Measurement, x)

    interp = DiffEqBase.LinearInterpolation(t,u)

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    return ProbODESolution{T,N,typeof(u),typeof(x),typeof(t),typeof(prob),typeof(alg),typeof(interp)}(
        u,x,t,prob,alg,dense,interp,retcode)
end


# Plot recipe for the solution: Plot with ribbon
@recipe function f(sol::AbstractProbODESolution; c=1.96)
    println("Hello plotting")
    stack(x) = copy(reduce(hcat, x)')
    values = map(u -> Measurements.value.(u), sol.u)
    uncertainties = map(u -> Measurements.uncertainty.(u), sol.u)
    ribbon := stack(uncertainties) * c
    return sol.t, stack(values)
end
