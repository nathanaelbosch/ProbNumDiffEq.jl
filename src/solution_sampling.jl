########################################################################################
# Sampling from a solution
########################################################################################
"""
    _rand(x::Gaussian{<:Vector,<:PSDMatrix}, n::Integer=1)

Sample from a Gaussian with a `ProbNumDiffEq.SquarerootMatrix` covariance.
Uses the existing covariance square root to make the sampling more stable.
"""
function _rand(x::SRGaussian, n::Integer=1)
    m, C = x.μ, x.Σ
    sample = m .+ C.R' * randn(size(C.R, 1), n)
    return sample
end

function sample_states(sol::ProbODESolution, n::Int=1)
    @assert sol.alg.smooth "sampling not implemented for non-smoothed posteriors"
    return sample_states(sol.t, sol.x_filt, sol.diffusions, sol.t, sol.cache, n)
end
function sample(sol::ProbODESolution, n::Int=1)
    sample_path = sample_states(sol, n)
    ys = cat(map(x -> (sol.cache.SolProj * x')', eachslice(sample_path; dims=3))...; dims=3)
    return ys
end
"""
    _sample_backward(μ, Σ, next_samples, A, Q)

Draw one sample of `x ~ N(μ, Σ)` conditioned on each column of `next_samples`, which are
samples of `A x + N(0, Q)`. `μ` is a vector, or a matrix with one column per sample.
"""
function _sample_backward(μ, Σ, next_samples, A, Q)
    # The gain `G` and the covariance of the conditional do not depend on the means
    x, G = smooth(
        Gaussian(μ[:, 1], Σ),
        Gaussian(next_samples[:, 1], PSDMatrix(zero(Σ.R))),
        A,
        Q,
    )
    return μ .+ G * (next_samples .- A * μ) .+ x.Σ.R' * randn(size(next_samples))
end
function sample_states(ts, xs, diffusions, difftimes, cache, n::Int=1)
    @assert length(diffusions) + 1 == length(difftimes)
    any(isnan, ts) && throw(ArgumentError("Cannot sample states at t=NaN"))

    sample_path = zeros(length(ts), length(xs[end].μ), n)
    sample_path[end, :, :] .= _rand(xs[end], n)
    for i in (length(xs)-1):-1:1
        i_diffusion = searchsortedlast(difftimes, ts[i]; lt=(<))
        diffusion = diffusions[min(i_diffusion, length(diffusions))]
        make_transition_matrices!(cache, ts[i+1] - ts[i])
        Qh = apply_diffusion(cache.Qh, diffusion)
        sample_path[i, :, :] .=
            _sample_backward(xs[i].μ, xs[i].Σ, sample_path[i+1, :, :], cache.Ah, Qh)
    end

    return sample_path
end
function dense_sample_states(sol::ProbODESolution, n::Int=1; density=1000)
    times = range(sol.t[1], sol.t[end], length=density)
    step_samples = sample_states(sol, n)

    # There is no data between two solver steps. So given the sample at the step before a
    # dense time, the filtering distribution at the dense time is the prediction from that
    # sample, and backward sampling conditions it on the next sample.
    samples = similar(step_samples, length(times), size(step_samples, 2), n)
    cache = sol.cache
    for i in reverse(eachindex(times))
        t = times[i]
        k = searchsortedlast(sol.t, t; lt=(<))
        if sol.t[k] == t
            samples[i, :, :] .= step_samples[k, :, :]
            continue
        end
        next_t, next_samples = if times[i+1] <= sol.t[k+1]
            times[i+1], samples[i+1, :, :]
        else
            sol.t[k+1], step_samples[k+1, :, :]
        end
        diffusion = sol.diffusions[k]
        make_transition_matrices!(cache, t - sol.t[k])
        predicted = cache.Ah * step_samples[k, :, :]
        Q_pred = apply_diffusion(cache.Qh, diffusion)
        make_transition_matrices!(cache, next_t - t)
        Qh = apply_diffusion(cache.Qh, diffusion)
        samples[i, :, :] .= _sample_backward(predicted, Q_pred, next_samples, cache.Ah, Qh)
    end

    return samples, times
end
function dense_sample(sol::ProbODESolution, n::Int=1; density=1000)
    samples, times = dense_sample_states(sol, n; density=density)
    ys = cat(map(x -> (sol.cache.SolProj * x')', eachslice(samples; dims=3))...; dims=3)
    return ys, times
end
