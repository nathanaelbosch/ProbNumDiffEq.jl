########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end

"""
    ProbODESolution

The probabilistic numerical ODE solution.

It contains filtering and smoothing state estimates which enables plots with uncertainties,
sampling, and dense evaluation.
"""
mutable struct ProbODESolution{
    T,N,uType,puType,uType2,DType,tType,rateType,xType,diffType,bkType,llType,P,A,IType,
    CType,DE,
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
    backward_kernels::bkType
    log_likelihood::llType
    prob::P
    alg::A
    interp::IType
    cache::CType
    dense::Bool
    tslocation::Int
    stats::DE
    retcode::ReturnCode.T
end
ProbODESolution{T,N}(
    u, pu, u_analytic, errors, t, k, x_filt, x_smooth, diffusions, backward_kernels,
    log_likelihood, prob, alg, interp, cache, dense, tslocation, stats, retcode,
) where {T,N} = ProbODESolution{
    T,N,typeof(u),typeof(pu),typeof(u_analytic),typeof(errors),typeof(t),typeof(k),
    typeof(x_filt),typeof(diffusions),typeof(backward_kernels),typeof(log_likelihood),
    typeof(prob),typeof(alg),typeof(interp),typeof(cache),typeof(stats),
}(
    u, pu, u_analytic, errors, t, k, x_filt, x_smooth, diffusions, backward_kernels,
    log_likelihood, prob, alg, interp, cache, dense, tslocation, stats, retcode,
)

function DiffEqBase.solution_new_retcode(sol::ProbODESolution{T,N}, retcode) where {T,N}
    return ProbODESolution{T,N}(
        sol.u, sol.pu, sol.u_analytic, sol.errors, sol.t, sol.k, sol.x_filt, sol.x_smooth,
        sol.diffusions, sol.backward_kernels, sol.log_likelihood, sol.prob, sol.alg,
        sol.interp, sol.cache, sol.dense, sol.tslocation, sol.stats, retcode,
    )
end

# Used to build the initial empty solution in OrdinaryDiffEq.__init
function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::AbstractEK,
    t,
    u;
    k=nothing,
    retcode=ReturnCode.Default,
    stats=nothing,
    dense=true,
    kwargs...,
)
    # By making an actual cache, interpolation can be written very closely to the solver
    cache = OrdinaryDiffEq.alg_cache(
        alg,
        prob.u0,
        recursivecopy(prob.u0),
        recursive_unitless_eltype(prob.u0),
        recursive_unitless_bottom_eltype(prob.u0),
        eltype(t),
        recursivecopy(prob.u0),
        recursivecopy(prob.u0),
        prob.f,
        t,
        eltype(prob.tspan)(1),
        nothing,
        prob.p,
        true,
        Val(isinplace(prob)),
    )

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    uElType = eltype(prob.u0)

    pu = StructArray{typeof(cache.pu_tmp)}(undef, 0)
    x_filt = StructArray{typeof(cache.x)}(undef, 0)
    x_smooth = copy(x_filt)

    backward_kernels = StructArray{typeof(cache.backward_kernel)}(undef, 0)

    interp = ODEFilterPosterior()

    if DiffEqBase.has_analytic(prob.f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol,real(uElType)}()
    else
        u_analytic = nothing
        errors = nothing
    end

    diffusion_prototype = cache.default_diffusion

    ll = zero(uElType)
    return ProbODESolution{T,N}(
        u, pu, u_analytic, errors, t, k, x_filt, x_smooth, typeof(diffusion_prototype)[],
        backward_kernels, ll, prob, alg, interp, cache,
        dense, 0, stats, retcode,
    )
end

function DiffEqBase.build_solution(
    sol::ProbODESolution{T,N},
    u_analytic,
    errors,
) where {T,N}
    return ProbODESolution{T,N}(
        sol.u, sol.pu, u_analytic, errors, sol.t, sol.k, sol.x_filt, sol.x_smooth,
        sol.diffusions, sol.backward_kernels, sol.log_likelihood, sol.prob, sol.alg,
        sol.interp, sol.cache, sol.dense, sol.tslocation, sol.stats, sol.retcode,
    )
end

########################################################################################
# Compat with classic ODE solutions, to enable analysis with DiffEqDevTools.jl
########################################################################################
"""
    MeanProbODESolution

The mean of a probabilistic numerical ODE solution.

Since it is the mean and does never return Gaussians, it can basically be treated as if it
were a classic ODE solution and is well-compatible with e.g. DiffEqDevtools.jl.
"""
mutable struct MeanProbODESolution{
    T,N,uType,uType2,DType,tType,rateType,P,A,IType,CType,DE,PSolType,
} <: DiffEqBase.AbstractODESolution{T,N,uType}
    u::uType
    u_analytic::uType2
    errors::DType
    t::tType
    k::rateType
    prob::P
    alg::A
    interp::IType
    cache::CType
    dense::Bool
    tslocation::Int
    stats::DE
    retcode::ReturnCode.T
    probsol::PSolType
end
MeanProbODESolution{T,N}(
    u, u_analytic, errs, t, k, prob, alg, interp, cache, dense, tsl, stats, retcode,
    probsol,
) where {T,N} = MeanProbODESolution{
    T,N,typeof(u),typeof(u_analytic),typeof(errs),typeof(t),typeof(k),typeof(prob),
    typeof(alg),typeof(interp),typeof(cache),typeof(stats),typeof(probsol)}(
    u, u_analytic, errs, t, k, prob, alg, interp, cache, dense, tsl, stats, retcode,
    probsol,
)

DiffEqBase.build_solution(sol::MeanProbODESolution{T,N}, u_analytic, errors) where {T,N} =
    MeanProbODESolution{T,N}(
        sol.u, u_analytic, errors, sol.t, sol.k, sol.prob, sol.alg, sol.interp, sol.cache,
        sol.dense, sol.tslocation, sol.stats, sol.retcode, sol.probsol)

function mean(sol::ProbODESolution{T,N}) where {T,N}
    return MeanProbODESolution{
        T,N,typeof(sol.u),typeof(sol.u_analytic),typeof(sol.errors),typeof(sol.t),
        typeof(sol.k),typeof(sol.prob),typeof(sol.alg),typeof(sol.interp),typeof(sol.cache),
        typeof(sol.stats),typeof(sol),
    }(
        sol.u, sol.u_analytic, sol.errors, sol.t, sol.k, sol.prob, sol.alg, sol.interp,
        sol.cache, sol.dense, sol.tslocation, sol.stats, sol.retcode, sol,
    )
end
(sol::MeanProbODESolution)(t::Real, args...) = mean(sol.probsol(t, args...))
(sol::MeanProbODESolution)(t::AbstractVector, args...) =
    DiffEqArray(sol.probsol(t, args...).u.μ, t)
DiffEqBase.calculate_solution_errors!(sol::ProbODESolution, args...; kwargs...) =
    DiffEqBase.calculate_solution_errors!(mean(sol), args...; kwargs...)

########################################################################################
# Dense Output
########################################################################################
abstract type AbstractODEFilterPosterior <: DiffEqBase.AbstractDiffEqInterpolation end
struct ODEFilterPosterior <: AbstractODEFilterPosterior end
DiffEqBase.interp_summary(interp::ODEFilterPosterior) =
    "ODE Filter Posterior"

(sol::ProbODESolution)(t::Real, args...) =
    sol.cache.SolProj * interpolate(
        t, sol.t, sol.x_filt, sol.x_smooth, sol.diffusions, sol.cache;
        smoothed=sol.alg.smooth)
(sol::ProbODESolution)(t::AbstractVector, args...) =
    DiffEqArray(StructArray(sol.(t, args...)), t)

function interpolate(
    tval::Real,
    t,
    x_filt,
    x_smooth,
    diffusions,
    cache;
    smoothed,
)
    @unpack d, q = cache

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
    make_transition_matrices!(cache, h1)

    # In principle the smoothing would look like this (without preconditioning):
    # @unpack Ah, Qh = posterior
    # Qh = apply_diffusion(Qh, diffusion)
    # goal_pred = predict(prev_rv, Ah, Qh)

    # To be numerically more stable, use the preconditioning:
    @unpack A, Q = cache
    P, PI = cache.P, cache.PI
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
    make_transition_matrices!(cache, h2)

    # In principle the smoothing would look like this (without preconditioning):
    # @unpack Ah, Qh = cache
    # Qh = apply_diffusion(Qh, diffusion)
    # goal_smoothed, _ = smooth(goal_pred, next_smoothed, Ah, Qh)

    # To be numerically more stable, use the preconditioning:
    @unpack A, Q = cache
    P, PI = cache.P, cache.PI
    goal_pred = P * goal_pred
    next_smoothed = P * next_smoothed
    Qh = apply_diffusion(Q, diffusion)
    goal_smoothed, _ = smooth(goal_pred, next_smoothed, A, Qh)
    goal_smoothed = PI * goal_smoothed

    return goal_smoothed
end
