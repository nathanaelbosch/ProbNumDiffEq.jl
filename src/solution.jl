########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,xType,tType,pType,P,A,S,IType,DE} <: AbstractProbODESolution{T,N,uType}
    u::uType
    x::xType
    t::tType
    proposals::pType
    prob::P
    alg::A
    solver::S
    dense::Bool
    interp::IType
    retcode::Symbol
    destats::DE
end

function make_Measurement(state, d)
    # @assert isdiag(state.Σ[1:d,1:d]) state.Σ[1:d,1:d]
    mean = state.μ[1:d]
    var = diag(state.Σ)[1:d]

    _min = minimum(var)
    if _min < 0
        @assert abs(_min) < 1e-16
        var .+= - _min
    end
    @assert all(var .>= 0)

    return mean .± sqrt.(var)
end

function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ODEFilter,
    t, x,
    proposals, solver;
    dense=false,
    retcode = :Default,
    destats = nothing,
    kwargs...)

    d = length(prob.u0)
    u = map(x -> make_Measurement(x, d), x)
    # u = map(s -> s.μ[1:d], x)

    interp = FilteringPosterior()

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    return ProbODESolution{T, N, typeof(u), typeof(x), typeof(t), typeof(proposals),
                           typeof(prob), typeof(alg), typeof(solver), typeof(interp), typeof(destats)}(
        u, x, t, proposals, prob, alg, solver, dense, interp, retcode, destats)
end

########################################################################################
# Dense Output
########################################################################################
struct FilteringPosterior <: DiffEqBase.AbstractDiffEqInterpolation end
DiffEqBase.interp_summary(::FilteringPosterior) = "Filtering Posterior" 

"""Just extrapolates with PREDICT so far"""
function (sol::ProbODESolution)(t::T) where T
    if t < sol.t[1]
        error("Invalid t")
    end 
    if t in sol.t
        idx = sum(sol.t .<= t)
        @assert sol.t[idx] == t
        return sol.u[idx]
    end
    
    # Extrapolate
    prev_idx = sum(sol.t .<= t)
    prev_rv = sol.x[prev_idx]
    m, P = prev_rv.μ, prev_rv.Σ
    h = t - sol.t[prev_idx]
    A, Q = sol.solver.dm.A(h), sol.solver.dm.Q(h)
    m_pred, P_pred = kf_predict(m, P, A, Q)

    pred_rv = Gaussian(m_pred, P_pred)
    d = sol.solver.d
    return make_Measurement(pred_rv, d)
end


########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution; c=1.96)
    stack(x) = collect(reduce(hcat, x)')
    values = map(u -> Measurements.value.(u), sol.u)
    uncertainties = map(u -> Measurements.uncertainty.(u), sol.u)
    ribbon := stack(uncertainties) * c
    return sol.t, stack(values)
end
