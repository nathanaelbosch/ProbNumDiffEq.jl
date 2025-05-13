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
    @unpack d, q = sol.cache
    sample_path = sample_states(sol, n)
    ys = cat(map(x -> (sol.cache.SolProj * x')', eachslice(sample_path; dims=3))...; dims=3)
    return ys
end
function sample_states(ts, xs, diffusions, difftimes, cache, n::Int=1)
    @assert length(diffusions) + 1 == length(difftimes)

    @unpack A, Q, d, q = cache
    D = d * (q + 1)

    x = xs[end]
    sample = _rand(x, n)
    @assert size(sample) == (D, n)

    sample_path = zeros(length(ts), D, n)
    sample_path[end, :, :] .= sample
    # @info "final value and samples" x.μ sample sample_path[end, :]

    for i in (length(xs)-1):-1:1
        dt = ts[i+1] - ts[i]

        i_diffusion = sum(difftimes .<= ts[i])
        diffusion = diffusions[min(i_diffusion, length(diffusions))]

        make_transition_matrices!(cache, dt)
        Ah, Qh = cache.Ah, cache.Qh
        Qh = apply_diffusion(Qh, diffusion)

        for j in 1:n
            sample_p = sample_path[i+1, :, j]
            x_prev_p = xs[i]

            prev_sample_p, _ =
                smooth(x_prev_p, Gaussian(sample_p, PSDMatrix(zeros(D, D))), Ah, Qh)

            # sample_path[i, :, j] .= PI*prev_sample_p.μ
            sample_path[i, :, j] .= _rand(prev_sample_p)[:]
        end
    end

    return sample_path
end
function dense_sample_states(sol::ProbODESolution, n::Int=1; density=1000)
    @assert sol.alg.smooth "sampling not implemented for non-smoothed posteriors"
    times = range(sol.t[1], sol.t[end], length=density)
    states = StructArray([
        interpolate(
            t,
            sol.t,
            sol.x_filt,
            sol.x_smooth,
            sol.diffusions,
            sol.cache;
            smoothed=sol.alg.smooth,
        )
        for t in times
    ])

    return sample_states(times, states, sol.diffusions, sol.t, sol.cache, n), times
end
function dense_sample(sol::ProbODESolution, n::Int=1; density=1000)
    samples, times = dense_sample_states(sol, n; density=density)
    @unpack d, q = sol.cache
    ys = cat(map(x -> (sol.cache.SolProj * x')', eachslice(samples; dims=3))...; dims=3)
    return ys, times
end
