########################################################################################
# Solution
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,IType,DE} <: AbstractProbODESolution{T,N,uType}
    u::uType
    pu
    u_analytic
    errors
    t
    k
    x
    diffusions
    prob
    alg
    interp::IType
    dense::Bool
    destats::DE
    retcode::Symbol
end
function DiffEqBase.solution_new_retcode(sol::ProbODESolution{T,N}, retcode) where {T,N}
    ProbODESolution{T, N, typeof(sol.u), typeof(sol.interp), typeof(sol.destats)}(
        sol.u, sol.pu, sol.u_analytic, sol.errors, sol.t, sol.k, sol.x, sol.diffusions,
        sol.prob, sol.alg, sol.interp, sol.dense, sol.destats, retcode,
    )
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
    uEltype = eltype(prob.u0)
    cov = zeros(uEltype, d, d)
    pu = StructArray{Gaussian{typeof(prob.u0), typeof(cov)}}(undef, 1)
    x = copy(pu)

    interp = GaussianODEFilterPosterior(alg, prob.u0)

    return ProbODESolution{T, N, typeof(u), typeof(interp), typeof(destats)}(
        u, pu, nothing, nothing, t, [], x, [], prob, alg, interp, dense, destats, retcode,
    )
end


function DiffEqBase.build_solution(sol::ProbODESolution{T,N}, u_analytic, errors) where {T,N}
    return ProbODESolution{T, N, typeof(sol.u), typeof(sol.interp), typeof(destats)}(
        sol.u, sol.pu, sol.p, u_analytic, errors, sol.x, sol.t, sol.diffusions, sol.prob,
        sol.alg, sol.solver, sol.dense, sol.interp, sol.retcode, sol.destats,
    )
end


########################################################################################
# Dense Output
########################################################################################
abstract type AbstractODEFilterPosterior <: DiffEqBase.AbstractDiffEqInterpolation end
struct GaussianODEFilterPosterior <: AbstractODEFilterPosterior
    d
    q
    A!
    Q!
    Ah
    Qh
    Precond
    InvPrecond
    smooth
end
function GaussianODEFilterPosterior(alg, u0)
    uElType = eltype(u0)
    d = length(u0)
    q = alg.order
    A!, Q! = ibm(d, q)
    Precond, InvPrecond = preconditioner(d, q)
    Ah = diagm(0=>ones(uElType, d*(q+1)))
    Qh = zeros(uElType, d*(q+1), d*(q+1))
    GaussianODEFilterPosterior(
        d, q, A!, Q!, Ah, Qh, Precond, InvPrecond, alg.smooth)
end
DiffEqBase.interp_summary(interp::GaussianODEFilterPosterior) = "Gaussian ODE Filter Posterior"

function (posterior::GaussianODEFilterPosterior)(tval::Real, t, x, diffusions)
    @unpack A!, Q!, Ah, Qh, d, q, Precond, InvPrecond = posterior

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
    diffmat = diffusions[minimum((idx, end))]

    # Extrapolate
    h1 = tval - prev_t
    P, PI = Precond(h1), InvPrecond(h1)
    A!(Ah, h1)
    Q!(Qh, h1)
    Qh .*= diffmat
    goal_pred = predict(P * prev_rv, Ah, Qh)
    goal_pred = PI * goal_pred

    if !posterior.smooth || tval >= t[end]
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
    Qh .*= diffmat

    goal_smoothed, _ = smooth(goal_pred, next_smoothed, Ah, Qh)

    return PI * goal_smoothed
end
function (sol::ProbODESolution)(t::Real, deriv::Val{N}=Val(0)) where {N}
    @unpack q, d = sol.interp
    E = kron([i==1 ? N+1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    return E * sol.interp(t, sol.t, sol.x, sol.diffusions)
end
(sol::ProbODESolution)(t::AbstractVector, deriv=Val(0)) = StructArray(sol.(t, deriv))



########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution; ribbon_width=1.96)
    times = range(sol.t[1], sol.t[end], length=1000)
    dense_post = sol(times)
    values = stack(mean(dense_post))
    stds = stack(std(dense_post))
    ribbon --> ribbon_width * stds
    xguide --> "t"
    yguide --> "u(t)"
    label --> hcat(["u$(i)(t)" for i in 1:length(sol.u[1])]...)
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
    sample(sol.t, sol.x, sol.diffusions, sol.interp, n)
end
function sample(ts, xs, diffusions, posterior, n::Int=1)

    @unpack A!, Q!, Ah, Qh, d, q, Precond, InvPrecond = posterior
    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    dim = d*(q+1)

    x = xs[end]
    sample = _rand(x, n)
    @assert size(sample) == (dim, n)

    sample_path = zeros(length(ts), dim, n)
    sample_path[end, :, :] .= sample
    # @info "final value and samples" x.μ sample sample_path[end, :]

    for i in length(xs)-1:-1:1
        dt = ts[i+1] - ts[i]

        i_diffusion = sum(ts .<= ts[i])
        diffmat = diffusions[i_diffusion]

        A!(Ah, dt)
        Q!(Qh, dt)
        Qh .*= diffmat
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
    states = StructArray(sol(times))

    sample(times, states, sol.diffusions, sol.interp, n), times
end
