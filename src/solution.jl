########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,puType,probType,uType2,DType,xType,tType,pType,P,A,S,IType,DE} <: AbstractProbODESolution{T,N,uType}
    u::uType                # array of non-probabilistic function values
    pu::puType              # array of Gaussians
    p::probType             # ODE posterior
    u_analytic::uType2
    errors::DType
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

    pu = StructArray(map(x -> E0 * x, x))
    u = pu.μ
    p = GaussianODEFilterPosterior(t, x, solver)
    interp = p

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    u_analytic = Vector{typeof(prob.u0)}()
    errors = Dict{Symbol,real(eltype(prob.u0))}()
    # errors[:final] = 0.0

    return ProbODESolution{T, N, typeof(u), typeof(pu), typeof(p), typeof(u_analytic), typeof(errors),
                           typeof(x), typeof(t), typeof(proposals),
                           typeof(prob), typeof(alg), typeof(solver), typeof(interp), typeof(destats)}(
        u, pu, p, u_analytic, errors, x, t, proposals, prob, alg, solver, dense, interp, retcode, destats)
end
function DiffEqBase.build_solution(sol::ProbODESolution{T,N}, u_analytic, errors) where {T,N}
    return ProbODESolution{
        T, N, typeof(sol.u), typeof(sol.pu), typeof(sol.p), typeof(u_analytic), typeof(errors),
        typeof(sol.x), typeof(sol.t), typeof(sol.proposals),
        typeof(sol.prob), typeof(sol.alg), typeof(sol.solver), typeof(sol.interp), typeof(sol.destats)}(
            sol.u, sol.pu, sol.p, u_analytic, errors, sol.x, sol.t, sol.proposals, sol.prob, sol.alg, sol.solver, sol.dense, sol.interp, sol.retcode, sol.destats)
end

########################################################################################
# Dense Output
########################################################################################
abstract type AbstractFilteringPosterior end
struct GaussianFilteringPosterior <: AbstractFilteringPosterior
    t
    x
    solver
end
function (posterior::GaussianFilteringPosterior)(tval::Real)
    @unpack t, x, solver = posterior

    @unpack A!, Q!, d, q, E0 = solver.constants
    @unpack Ah, Qh = solver.cache

    if tval < t[1]
        error("Invalid t<t0")
    end
    if tval in t
        idx = sum(t .<= tval)
        @assert t[idx] == tval
        return x[idx]
    end

    idx = sum(t .<= tval)
    prev_rv, next_rv = x[idx:idx+1]
    prev_t, next_t = t[idx:idx+1]
    σ² = 1.0

    # Extrapolate
    m, P = prev_rv.μ, prev_rv.Σ
    h1 = tval - prev_t
    A!(Ah, h1)
    Q!(Qh, h1, σ²)
    m_pred, P_pred = kf_predict(m, P, Ah, Qh)

    pred_rv = Gaussian(m_pred, P_pred)

    if !solver.smooth || tval > t[end]
        return pred_rv
    end

    # Smooth
    h2 = next_t - tval
    A!(Ah, h2)
    Q!(Qh, h2, σ²)
    m_pred_next, P_pred_next = kf_predict(m_pred, P_pred, Ah, Qh)

    m_smoothed, P_smoothed = kf_smooth(
        m_pred, P_pred, m_pred_next, P_pred_next, next_rv.μ, next_rv.Σ, Ah, Qh)
    smoothed_rv = Gaussian(m_smoothed, P_smoothed)

    return smoothed_rv

end
(posterior::GaussianFilteringPosterior)(tvals::AbstractVector) = StructArray(posterior.(tvals))
Base.getindex(post::GaussianFilteringPosterior, key) = post.x[key]


abstract type AbstractODEFilterPosterior <: DiffEqBase.AbstractDiffEqInterpolation end
struct GaussianODEFilterPosterior <: AbstractODEFilterPosterior
    filtering_posterior::GaussianFilteringPosterior
    proj
end

function GaussianODEFilterPosterior(t, x, solver)
    GaussianODEFilterPosterior(
        GaussianFilteringPosterior(t, x, solver),
        solver.constants.E0)
end
DiffEqBase.interp_summary(interp::GaussianODEFilterPosterior) = "Gaussian ODE Filter Posterior"
Base.getindex(post::GaussianODEFilterPosterior, key) = post.proj * post.filtering_posterior[key]

function (ode_posterior::GaussianODEFilterPosterior)(t::Real)
    return ode_posterior.proj * ode_posterior.filtering_posterior(t)
end
function (ode_posterior::GaussianODEFilterPosterior)(tvec::AbstractVector)
    return StructArray([ode_posterior.proj * ode_posterior.filtering_posterior(t)
                        for t in tvec])
end

(sol::ProbODESolution)(t::Real) = sol.p(t).μ
(sol::ProbODESolution)(t::AbstractVector) = DiffEqBase.DiffEqArray(sol.(t), t)




########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution; c=1.96)
    stack(x) = collect(reduce(hcat, x)')
    values = stack(sol.pu.μ)
    stds = stack([sqrt.(diag(cov)) for cov in sol.pu.Σ])
    ribbon := c * stds
    return sol.t, values
end
