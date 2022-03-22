########################################################################################
# Sampling from a solution
########################################################################################
"""
    _rand(x::Gaussian{<:Vector,<:SRMatrix}, n::Integer=1)

Sample from a Gaussian with a `ProbNumDiffEq.SquarerootMatrix` covariance.
Uses the existing covariance square root to make the sampling more stable.
"""
function _rand(x::SRGaussian, n::Integer=1)
    m, C = x.μ, x.Σ
    sample = m .+ C.R' * randn(length(m), n)
    return sample
end

function sample_states(sol::ProbODESolution, n::Int=1)
    @assert sol.interp.smooth "sampling not implemented for non-smoothed posteriors"
    return sample_states(sol.t, sol.x_filt, sol.diffusions, sol.t, sol.interp, n)
end
function sample(sol::ProbODESolution, n::Int=1)
    d, q = sol.interp.d, sol.interp.q
    sample_path = sample_states(sol, n)
    return sample_path[:, 1:q+1:d*(q+1), :]
end
function sample_states(ts, xs, diffusions, difftimes, posterior, n::Int=1)
    @assert length(diffusions) + 1 == length(difftimes)

    @unpack A, Q, d, q = posterior
    D = d * (q + 1)

    x = xs[end]
    sample = _rand(x, n)
    @assert size(sample) == (D, n)

    sample_path = zeros(length(ts), D, n)
    sample_path[end, :, :] .= sample
    # @info "final value and samples" x.μ sample sample_path[end, :]

    for i in length(xs)-1:-1:1
        dt = ts[i+1] - ts[i]

        i_diffusion = sum(difftimes .<= ts[i])
        diffusion = diffusions[i_diffusion]

        Qh = apply_diffusion(Q, diffusion)
        make_preconditioners!(posterior, dt)
        P, PI = posterior.P, posterior.PI

        for j in 1:n
            sample_p = P * sample_path[i+1, :, j]
            x_prev_p = P * xs[i]

            prev_sample_p, _ =
                smooth(x_prev_p, Gaussian(sample_p, SRMatrix(zeros(D, D))), A, Qh)

            # sample_path[i, :, j] .= PI*prev_sample_p.μ
            sample_path[i, :, j] .= PI * _rand(prev_sample_p)[:]
        end
    end

    return sample_path
end
function dense_sample_states(sol::ProbODESolution, n::Int=1; density=1000)
    @assert sol.interp.smooth "sampling not implemented for non-smoothed posteriors"
    times = range(sol.t[1], sol.t[end], length=density)
    states = StructArray([
        sol.interp(t, sol.t, sol.x_filt, sol.x_smooth, sol.diffusions; smoothed=false)
        for t in times
    ])

    return sample_states(times, states, sol.diffusions, sol.t, sol.interp, n), times
end
function dense_sample(sol::ProbODESolution, n::Int=1; density=1000)
    samples, times = dense_sample_states(sol, n; density=density)
    d, q = sol.interp.d, sol.interp.q
    return samples[:, 1:q+1:d*(q+1), :], times
end
