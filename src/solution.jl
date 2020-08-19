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

function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ODEFilter,
    t, x,
    proposals, solver;
    dense=false,
    retcode = :Default,
    destats = nothing,
    kwargs...)
    @unpack d, q, E0 = solver.constants

    u = StructArray(map(x -> E0 * x, x))

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
    @unpack A!, Q!, d, q, E0 = sol.solver.constants
    @unpack Ah, Qh = sol.solver.cache

    if t < sol.t[1]
        error("Invalid t<t0")
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
    A!(Ah, h)
    Q!(Qh, h)
    m_pred, P_pred = kf_predict(m, P, Ah, Qh)

    pred_rv = Gaussian(m_pred, P_pred)

    if !sol.solver.smooth || t > sol.t[end]
        return E0 * pred_rv
    end

    # Smooth
    next_rv = sol.x[prev_idx+1]
    h = sol.t[prev_idx+1] - t
    m_pred_next, P_pred_next = kf_predict(m, P, Ah, Qh)
    m_smoothed, P_smoothed = kf_smooth(
        m_pred, P_pred, m_pred_next, P_pred_next, next_rv.μ, next_rv.Σ, Ah, Qh)
    smoothed_rv = Gaussian(m_smoothed, P_smoothed)
    return E0 * smoothed_rv
end


########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution; c=1.96)
    stack(x) = collect(reduce(hcat, x)')
    values = stack(sol.u.μ)
    stds = stack([sqrt.(diag(cov)) for cov in sol.u.Σ])
    ribbon := c * stds
    return sol.t, values
end
