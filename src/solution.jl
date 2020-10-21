########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,puType,probType,uType2,DType,xType,tType,sType,P,A,S,IType,DE} <: AbstractProbODESolution{T,N,uType}
    u::uType                # array of non-probabilistic function values
    pu::puType              # array of Gaussians
    p::probType             # ODE posterior
    u_analytic::uType2
    errors::DType
    x::xType
    t::tType
    diffusions::sType
    prob::P
    alg::A
    solver::S
    dense::Bool
    interp::IType
    retcode::Symbol
    destats::DE
end
function solution_new_retcode(sol::ProbODESolution{T,N}, retcode) where {T,N}
    ProbODESolution{T, N, typeof(sol.u), typeof(sol.pu), typeof(sol.p),
                    typeof(sol.u_analytic), typeof(sol.errors), typeof(sol.x),
                    typeof(sol.t), typeof(sol.diffusions),
                    typeof(sol.prob), typeof(sol.alg), typeof(sol.solver),
                    typeof(sol.interp), typeof(sol.destats)}(
                        sol.u, sol.pu, sol.p, sol.u_analytic, sol.errors, sol.x, sol.t,
                        sol.diffusions, sol.prob, sol.alg, sol.solver, sol.dense,
                        sol.interp, retcode, sol.destats)
end

function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ODEFilter,
    t, x, diffusions,
    solver;
    retcode = :Default,
    destats = nothing,
    timeseries_errors = length(x)>2,
    dense = true, dense_errors = dense,
    calculate_error = true,
    kwargs...)
    @unpack d, q, E0 = solver.cache

    @assert length(t) == length(x) == (length(diffusions)+1)

    x = StructArray(x)
    pu = StructArray(map(x -> E0 * x, x))
    u = pu.μ
    p = GaussianODEFilterPosterior(t, x, diffusions, solver)
    interp = p

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    u_analytic = nothing
    Vector{typeof(prob.u0)}()
    if DiffEqBase.has_analytic(prob.f)
        u_analytic = Vector{typeof(prob.u0)}()
        errors = Dict{Symbol,real(eltype(prob.u0))}()

        sol = ProbODESolution{
            T, N, typeof(u), typeof(pu), typeof(p),
            typeof(u_analytic), typeof(errors),
            typeof(x), typeof(t), typeof(diffusions),
            typeof(prob), typeof(alg), typeof(solver), typeof(interp), typeof(destats)}(
                u, pu, p, u_analytic, errors, x, t, diffusions, prob, alg, solver, dense, interp, retcode, destats)

        if calculate_error
            DiffEqBase.calculate_solution_errors!(sol;
                timeseries_errors=timeseries_errors,
                dense_errors=dense_errors)
        end

        return sol
    else
        return ProbODESolution{
            T, N, typeof(u), typeof(pu), typeof(p), Nothing, Nothing,
            typeof(x), typeof(t), typeof(diffusions),
            typeof(prob), typeof(alg), typeof(solver), typeof(interp), typeof(destats)}(
                u, pu, p, nothing, nothing, x, t, diffusions, prob, alg, solver, dense, interp, retcode, destats)
    end
end

function DiffEqBase.build_solution(sol::ProbODESolution{T,N}, u_analytic, errors) where {T,N}
    return ProbODESolution{
        T, N, typeof(sol.u), typeof(sol.pu), typeof(sol.p), typeof(u_analytic), typeof(errors),
        typeof(sol.x), typeof(sol.t), typeof(sol.diffusions),
        typeof(sol.prob), typeof(sol.alg), typeof(sol.solver), typeof(sol.interp), typeof(sol.destats)}(
            sol.u, sol.pu, sol.p, u_analytic, errors, sol.x, sol.t, sol.diffusions, sol.prob, sol.alg, sol.solver, sol.dense, sol.interp, sol.retcode, sol.destats)
end

########################################################################################
# Dense Output
########################################################################################
abstract type AbstractFilteringPosterior end
struct GaussianFilteringPosterior <: AbstractFilteringPosterior
    t
    x
    diffusions
    solver
end
function (posterior::GaussianFilteringPosterior)(tval::Real)
    @unpack t, x, diffusions, solver = posterior

    @unpack A!, Q!, d, q, E0, Precond, InvPrecond = solver.cache
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
    prev_t = t[idx]
    prev_rv = x[idx]
    σ² = diffusions[minimum((idx, end))]

    # Extrapolate
    h1 = tval - prev_t
    P, PI = Precond(h1), InvPrecond(h1)
    A!(Ah, h1)
    Q!(Qh, h1)
    Qh .*= σ²
    goal_pred = predict(P * prev_rv, Ah, Qh)
    goal_pred = PI * goal_pred

    if !solver.smooth || tval >= t[end]
        return goal_pred
    end

    next_t = t[idx+1]
    next_smoothed = x[idx+1]

    # Smooth
    h2 = next_t - tval
    P, PI = Precond(h2), InvPrecond(h2)
    goal_pred = P * goal_pred
    next_smoothed = P * next_smoothed
    A!(Ah, h2)
    Q!(Qh, h2)
    Qh .*= σ²

    goal_smoothed, _ = smooth(goal_pred, next_smoothed, Ah, Qh)

    return PI * goal_smoothed

end
(posterior::GaussianFilteringPosterior)(tvals::AbstractVector) = StructArray(posterior.(tvals))
Base.getindex(post::GaussianFilteringPosterior, key) = post.x[key]


abstract type AbstractODEFilterPosterior <: DiffEqBase.AbstractDiffEqInterpolation end
struct GaussianODEFilterPosterior <: AbstractODEFilterPosterior
    filtering_posterior::GaussianFilteringPosterior
    proj
end

function GaussianODEFilterPosterior(t, x, diffusions, solver)
    GaussianODEFilterPosterior(
        GaussianFilteringPosterior(t, x, diffusions, solver),
        solver.cache.E0)
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
    times = range(sol.t[1], sol.t[end], length=1000)
    # values = stack(sol.u)
    pus = StructArray(sol.p.(times))
    values = stack(pus.μ)
    # vars = stack(diag.(sol.pu.Σ))
    vars = stack(diag.(pus.Σ))
    stds = sqrt.(vars)
    ribbon --> c * stds
    xguide --> "t"
    yguide --> "y(t)"
    return times, values
end


########################################################################################
# Sampling from a solution
########################################################################################
# function _rand(x::Gaussian, n::Int=1)
#     chol = cholesky(Symmetric(x.Σ))
#     sample = x.μ .+ chol.L*randn(length(x.μ), n)
#     return sample
# end

function get_zero_cross_indices(C)
    bad_idx = []
    D = size(C)[1]
    for i in 1:D
        if all(C[i, :] .< eps(eltype(C))) && all(C[:, i] .< eps(eltype(C)))
            push!(bad_idx, i)
        end
    end
    return bad_idx
end

"""Helper function to sample from our covariances, which often have a "cross" of zeros
For the 0-cov entries the outcome of the sampling is deterministic!"""
function _rand(x::Gaussian, n::Int=1)
    m, C = x.μ, x.Σ

    try
        chol = cholesky(Symmetric(C))
        sample = m .+ chol.L*randn(length(m), n)
        return sample
    catch e
        @warn "Cholesky failed; Try to sample more manually" e
        bad_idx = get_zero_cross_indices(C)

        D = length(m)
        @assert all(C[bad_idx, :] .< eps(eltype(C)))
        @assert all(C[:, bad_idx] .< eps(eltype(C)))
        reduced_idx = setdiff(1:D, bad_idx)
        reduced_x = Gaussian(m[reduced_idx], Symmetric(C[reduced_idx, reduced_idx]))
        reduced_sample = rand(reduced_x)

        sample = reduced_sample
        for i in bad_idx
            insert!(sample, i, m[i])
        end
        @assert sample[bad_idx] == m[bad_idx]

        return sample
    end
end


function sample_back(x_curr::Gaussian, x_next_sample::AbstractVector, Ah::AbstractMatrix, Qh::AbstractMatrix, PI=I)
    m_p, P_p = Ah*x_curr.μ, Ah*x_curr.Σ*Ah' + Qh
    P_p_inv = inv(Symmetric(P_p))
    Gain = x_curr.Σ * Ah' * P_p_inv

    m = x_curr.μ + Gain * (x_next_sample - m_p)

    P = ((I - Gain*Ah) * x_curr.Σ * (I - Gain*Ah)'
         + Gain * Qh * Gain')

    assert_nonnegative_diagonal(P)
    return Gaussian(m, P)
end


function sample(sol::ProbODESolution, n::Int=1)
    sample(sol.t, sol.x, sol.solver, n)
end
function sample(ts, xs, solver, n::Int=1)

    @unpack A!, Q!, d, q, E0, Precond, InvPrecond = solver.cache
    @unpack Ah, Qh = solver.cache
    dim = d*(q+1)

    x = xs[end]
    sample = _rand(x, n)
    @assert size(sample) == (dim, n)

    sample_path = zeros(length(ts), dim, n)
    sample_path[end, :, :] .= sample
    # @info "final value and samples" x.μ sample sample_path[end, :]

    for i in length(xs)-1:-1:1
        dt = ts[i+1] - ts[i]

        i_diffusion = sum(solver.times .<= ts[i])
        σ² = solver.sol.diffusions[i_diffusion]

        A!(Ah, dt)
        Q!(Qh, dt)
        Qh .*= σ²
        P, PI = Precond(dt), InvPrecond(dt)

        for j in 1:n
            sample_p = P*sample_path[i+1, :, j]
            x_prev_p = P*xs[i]

            prev_sample_p = sample_back(x_prev_p, sample_p, Ah, Qh, PI)

            # sample_path[i, :, j] .= PI*prev_sample_p.μ
            sample_path[i, :, j] .= PI*_rand(prev_sample_p)[:]
        end
    end

    return sample_path[:, 1:d, :]
end
function dense_sample(sol::ProbODESolution, n::Int=1)
    times = range(sol.t[1], sol.t[end], length=1000)
    states = StructArray(sol.p.filtering_posterior(times))

    sample(times, states, sol.solver, n), times
end
