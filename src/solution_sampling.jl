########################################################################################
# Sampling from a solution
########################################################################################
"""Helper function to sample from our covariances, which often have a "cross" of zeros
For the 0-cov entries the outcome of the sampling is deterministic!"""
function _rand(x::Gaussian, n::Int=1)
    m, C = x.μ, x.Σ
    @assert C isa SRMatrix

    sample = m .+ C.squareroot*randn(length(m), n)
    return sample
end


function sample_back(x_curr::SRGaussian, x_next_sample::AbstractVector, Ah::AbstractMatrix, Qh::SRMatrix, PI=I)
    m_p, P_p = Ah*x_curr.μ, Ah*x_curr.Σ*Ah' + Qh
    P_p_inv = inv(Symmetric(P_p))
    Gain = x_curr.Σ * Ah' * P_p_inv

    m = x_curr.μ + Gain * (x_next_sample - m_p)

    # P = X_A_Xt(x_curr.Σ, (I - Gain*Ah)) + X_A_Xt(Qh, Gain)
    _R = [x_curr.Σ.squareroot' * (I - Gain*Ah)'
          Qh.squareroot' * Gain']
    _, P_L = qr(_R)
    P = SRMatrix(P_L)

    assert_nonnegative_diagonal(P)
    return Gaussian(m, P)
end


function sample(sol::ProbODESolution, n::Int=1)
    sample(sol.t, sol.x_filt, sol.diffusions, sol.t, sol.interp, n)
end
function sample(ts, xs, diffusions, difftimes, posterior, n::Int=1)

    @unpack A, Q, d, q, Precond = posterior
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

        i_diffusion = sum(difftimes .<= ts[i])
        diffusion = diffusions[i_diffusion]

        Qh = apply_diffusion(Q, diffusion)
        P = Precond(dt)
        PI = inv(P)

        for j in 1:n
            sample_p = P*sample_path[i+1, :, j]
            x_prev_p = P*xs[i]

            prev_sample_p = sample_back(x_prev_p, sample_p, A, Qh, PI)

            # sample_path[i, :, j] .= PI*prev_sample_p.μ
            sample_path[i, :, j] .= PI*_rand(prev_sample_p)[:]
        end
    end

    return sample_path[:, 1:d, :]
end
function dense_sample(sol::ProbODESolution, n::Int=1)
    times = range(sol.t[1], sol.t[end], length=1000)
    states = StructArray([sol.interp(t, sol.t, sol.x, sol.diffusions) for t in times])

    sample(times, states, sol.diffusions, sol.t, sol.interp, n), times
end
