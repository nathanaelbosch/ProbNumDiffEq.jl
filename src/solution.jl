########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
mutable struct ProbODESolution{
    T, N, uType, puType, uType2, DType, tType, rateType, xType, diffType, llType, P, A, IType, DE
} <: AbstractProbODESolution{T,N,uType}
    u::uType
    pu::puType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    x_filt::xType
    x_smooth::xType
    diffusions::diffType
    log_likelihood::llType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    destats::DE
    retcode::Symbol
end
ProbODESolution{T, N}(
    u, pu, u_analytic, errors, t, k, x_filt, x_smooth, diffusions, log_likelihood, prob, alg, interp,
    dense, tslocation, destats, retcode) where {T, N} =
        ProbODESolution{
            T, N, typeof(u), typeof(pu), typeof(u_analytic), typeof(errors), typeof(t),
            typeof(k), typeof(x_filt), typeof(diffusions), typeof(log_likelihood), typeof(prob),
            typeof(alg), typeof(interp), typeof(destats)
        }(
            u, pu, u_analytic, errors, t, k, x_filt, x_smooth, diffusions, log_likelihood, prob, alg,
            interp, dense, tslocation, destats, retcode)

function DiffEqBase.solution_new_retcode(sol::ProbODESolution{T,N}, retcode) where {T,N}
    ProbODESolution{T, N}(
        sol.u, sol.pu, sol.u_analytic, sol.errors, sol.t, sol.k, sol.x_filt, sol.x_smooth,
        sol.diffusions, sol.log_likelihood, sol.prob, sol.alg, sol.interp,
        sol.dense, sol.tslocation, sol.destats, retcode)
end

# Used to build the initial empty solution in OrdinaryDiffEq.__init
function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::GaussianODEFilter,
    t, u;
    k = nothing,
    retcode = :Default,
    destats = nothing,
    dense = true,
    kwargs...)

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    d = length(prob.u0)
    uElType = eltype(prob.u0)
    D = d
    pu_cov = alg isa EK0 && !(prob.f isa DynamicalODEFunction) ?
        SRMatrix(zeros(uElType, d, D), Diagonal(zeros(uElType, d, d))) :
        SRMatrix(zeros(uElType, d, D))
    x_cov = SRMatrix(zeros(uElType, d, d))
    pu = StructArray{Gaussian{Vector{uElType}, typeof(pu_cov)}}(undef, 0)
    x_filt = StructArray{Gaussian{Vector{uElType}, typeof(x_cov)}}(undef, 0)
    x_smooth = copy(x_filt)

    interp = GaussianODEFilterPosterior(alg, prob.u0)

    if DiffEqBase.has_analytic(prob.f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol, real(uElType)}()
    else
        u_analytic = nothing
        errors = nothing
    end

    ll = zero(uElType)
    return ProbODESolution{T, N}(
        u, pu, u_analytic, errors, t, [], x_filt, x_smooth, [], ll, prob, alg, interp, dense, 0, destats, retcode,
    )
end


function DiffEqBase.build_solution(sol::ProbODESolution{T,N}, u_analytic, errors) where {T,N}
    return ProbODESolution{T, N}(
        sol.u, sol.pu, u_analytic, errors, sol.t, sol.k, sol.x_filt, sol.x_smooth,
        sol.diffusions, sol.log_likelihood, sol.prob,
        sol.alg, sol.interp, sol.dense, sol.tslocation, sol.destats, sol.retcode,
    )
end


########################################################################################
# Compat with classic ODE solutions, to enable analysis with DiffEqDevTools.jl
########################################################################################
"""Mean of a probabilistic ODE solution

Used for compatibility with classic ODE solutions, e.g. in `DiffEqDevTools.appxtrue`."""
mutable struct MeanProbODESolution{
    T, N, uType, uType2, DType, tType, rateType, P, A, IType, DE, PSolType
} <: DiffEqBase.AbstractODESolution{T,N,uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::IType
    dense::Bool
    tslocation::Int
    destats::DE
    retcode::Symbol
    probsol::PSolType
end
function mean(sol::ProbODESolution{T,N}) where {T,N}
    return MeanProbODESolution{
        T, N, typeof(sol.u), typeof(sol.u_analytic), typeof(sol.errors),
        typeof(sol.t), typeof(sol.k), typeof(sol.prob), typeof(sol.alg),
        typeof(sol.interp), typeof(sol.destats), typeof(sol)
    }(
        sol.u, sol.u_analytic, sol.errors, sol.t, sol.k, sol.prob,
        sol.alg, sol.interp, sol.dense, sol.tslocation, sol.destats,
        sol.retcode, sol)
end
(sol::MeanProbODESolution)(t::Real, deriv::Val{N}=Val(0)) where {N} =
    mean(sol.probsol(t, deriv))
(sol::MeanProbODESolution)(t::AbstractVector, deriv=Val(0)) =
    DiffEqArray(mean(sol.probsol(t, deriv).u), t)
DiffEqBase.calculate_solution_errors!(sol::ProbODESolution, args...; kwargs...) =
    DiffEqBase.calculate_solution_errors!(mean(sol), args...; kwargs...)




########################################################################################
# Dense Output
########################################################################################
abstract type AbstractODEFilterPosterior <: DiffEqBase.AbstractDiffEqInterpolation end
struct GaussianODEFilterPosterior{SPType, AType, QType, PType} <: AbstractODEFilterPosterior
    d::Int
    q::Int
    SolProj::SPType
    A::AType
    Q::QType
    P::PType
    PI::PType
    smooth::Bool
end
set_smooth(p::GaussianODEFilterPosterior) = GaussianODEFilterPosterior(
    p.d, p.q, p.SolProj, p.A, p.Q, p.P, p.PI, true)
function GaussianODEFilterPosterior(alg, u0)
    uElType = eltype(u0)
    d = u0 isa ArrayPartition ? length(u0) ÷ 2 : length(u0)
    q = alg.order
    D = d*(q+1)

    Proj = projection(d, q, uElType)
    SolProj = u0 isa ArrayPartition ? [Proj(1); Proj(0)] : Proj(0)

    A, Q = ibm(d, q, uElType)
    P, PI = init_preconditioner(d, q, uElType)
    GaussianODEFilterPosterior(
        d, q, SolProj, A, Q, P, PI, false)
end
DiffEqBase.interp_summary(interp::GaussianODEFilterPosterior) = "Gaussian ODE Filter Posterior"

function (posterior::GaussianODEFilterPosterior)(
    tval::Real, t, x_filt, x_smooth, diffusions; smoothed=posterior.smooth)
    @unpack A, Q, d, q = posterior

    if tval < t[1]
        error("Invalid t<t0")
    end
    if tval in t
        idx = sum(t .<= tval)
        @assert t[idx] == tval
        return smoothed ? x_smooth[idx] : x_filt[idx]
    end

    idx = sum(t .<= tval)
    prev_t = t[idx]
    prev_rv = x_filt[idx]
    diffusion = diffusions[minimum((idx, end))]

    # Extrapolate
    h1 = tval - prev_t
    make_preconditioners!(posterior, h1)
    P, PI = posterior.P, posterior.PI
    Qh = apply_diffusion(Q, diffusion)
    goal_pred = predict(P * prev_rv, A, Qh)
    goal_pred = PI * goal_pred

    if !smoothed || tval >= t[end]
        return goal_pred
    end

    @assert length(x_filt) == length(x_smooth)
    next_t = t[idx+1]
    next_smoothed = x_smooth[idx+1]

    # Smooth
    h2 = next_t - tval
    make_preconditioners!(posterior, h2)
    P, PI = posterior.P, posterior.PI
    goal_pred = P * goal_pred
    next_smoothed = P * next_smoothed
    Qh = apply_diffusion(Q, diffusion)

    goal_smoothed, _ = smooth(goal_pred, next_smoothed, A, Qh)

    return PI * goal_smoothed
end
function (sol::ProbODESolution)(t::Real, deriv::Val{N}=Val(0)) where {N}
    @unpack q, d = sol.interp
    return sol.interp.SolProj * sol.interp(t, sol.t, sol.x_filt, sol.x_smooth, sol.diffusions)
end
(sol::ProbODESolution)(t::AbstractVector, deriv=Val(0)) = DiffEqArray(StructArray(sol.(t, deriv)), t)
