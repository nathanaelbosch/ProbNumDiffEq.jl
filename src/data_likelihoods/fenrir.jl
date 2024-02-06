"""
$(TYPEDSIGNATURES)

Compute the Fenrir [tronarp22fenrir](@cite) approximate negative log-likelihood (NLL) of the data.

This is a convenience function that
1. Solves the ODE with a `ProbNumDiffEq.EK1` of the specified order and with a diffusion
   as provided by the `diffusion_var` argument, and
2. Fits the ODE posterior to the data via Kalman filtering and thereby computes the
   log-likelihood of the data on the way.

You can control the step-size behaviour of the solver as you would for a standard ODE solve,
but additionally the solver always steps through the `data.t` locations by adding them to
`tstops`.

You can also choose steps adaptively by setting `adaptive=true`, but this is not well-tested
so use at your own risk!

# Arguments
- `prob::SciMLBase.AbstractODEProblem`: the initial value problem of interest
- `alg::AbstractEK`: the probabilistic ODE solver to be used; use `EK1` for best results.
- `data::NamedTuple{(:t, :u)}`: the data to be fitted
- `observation_matrix::Union{AbstractMatrix,UniformScaling}`:
  the matrix which maps the ODE state to the measurements; typically a projection matrix
- `observation_noise_cov::Union{Number,AbstractMatrix}`: the scalar observation noise variance

# Reference
* [tronarp22fenrir](@cite) Tronarp et al, "Fenrir: Physics-Enhanced Regression for Initial Value Problems", ICML (2022)
"""
function fenrir_data_loglik(
    prob::SciMLBase.AbstractODEProblem,
    alg::AbstractEK,
    args...;
    # observation model
    observation_matrix=I,
    observation_noise_cov::Union{Number,AbstractMatrix},
    # data
    data::NamedTuple{(:t, :u)},
    kwargs...
)
    if !alg.smooth
        throw(ArgumentError("fenrir only works with smoothing. Set `smooth=true`."))
    end
    tstops = union(data.t, get(kwargs, :tstops, []))

    integ = init(prob, alg, args...; tstops, kwargs...)

    T = prob.tspan[2] - prob.tspan[1]
    step!(integ, T, false) # basically `solve!` but this prevents smoothing
    sol = integ.sol

    if sol.retcode != :Success && sol.retcode != :Default
        @error "The PN ODE solver did not succeed!" sol.retcode
        return -Inf * one(eltype(integ.p))
    end

    # Fit the ODE solution / PN posterior to the provided data; this is the actual Fenrir
    NLL, times, states =
        fit_pnsolution_to_data!(sol, observation_noise_cov, data; proj=observation_matrix)
    # u_probs = project_to_solution_space!(sol.pu, states, sol.cache.SolProj)

    return -NLL
end

function fit_pnsolution_to_data!(
    sol::AbstractProbODESolution,
    observation_noise_var::Real,
    data::NamedTuple{(:t, :u)};
    proj=I,
)
    @unpack cache, backward_kernels = sol
    @unpack A, Q, x_tmp, x_tmp2, m_tmp, C_DxD, C_3DxD = cache

    E = length(data.u[1])
    P = length(sol.prob.p)

    NLL = zero(eltype(sol.prob.p))

    measurement_cache = get_lowerdim_measurement_cache(m_tmp, E)

    x_posterior = copy(sol.x_filt) # the object to be filled
    state2data_projmat = proj * cache.SolProj
    observation_noise = Diagonal(observation_noise_var .* ones(E))
    ZERO_DATA = zeros(E)

    # First update on the last data point
    if sol.t[end] in data.t
        NLL += compute_nll_and_update!(
            x_posterior[end],
            data.u[end],
            state2data_projmat,
            observation_noise,
            measurement_cache,
            ZERO_DATA,
            cache,
        )
    end

    # Now iterate backwards
    data_idx = length(data.u) - 1
    for i in length(x_posterior)-1:-1:1
        # logic closely related to ProbNumDiffEq.jl's `smooth_solution!`
        if sol.t[i] == sol.t[i+1]
            copy!(x_posterior[i], x_posterior[i+1])
            continue
        end

        K = backward_kernels[i]
        marginalize!(x_posterior[i], x_posterior[i+1], K; C_DxD, C_3DxD)

        if data_idx > 0 && sol.t[i] == data.t[data_idx]
            NLL += compute_nll_and_update!(
                x_posterior[i],
                data.u[data_idx],
                state2data_projmat,
                observation_noise,
                measurement_cache,
                ZERO_DATA,
                cache,
            )
            data_idx -= 1
        end
    end
    @assert data_idx == 0 # to make sure we went through all the data

    return NLL, sol.t, x_posterior
end

function get_lowerdim_measurement_cache(m_tmp, E)
    _z, _S = m_tmp
    return Gaussian(view(_z, 1:E), PSDMatrix(view(_S.R, :, 1:E)))
end

function measure!(x, H, R, m_tmp)
    z, S = m_tmp
    _matmul!(z, H, x.μ)
    fast_X_A_Xt!(S, x.Σ, H)
    _S = Matrix(S) .+= R
    return Gaussian(z, Symmetric(_S))
end

function update!(
    x_out::SRGaussian,
    x_pred::SRGaussian,
    measurement::Gaussian,
    R::Diagonal,
    H::AbstractMatrix,
    K1_cache::AbstractMatrix,
    K2_cache::AbstractMatrix,
    M_cache::AbstractMatrix,
    C_dxd::AbstractMatrix,
)
    z, S = measurement.μ, measurement.Σ
    m_p, P_p = x_pred.μ, x_pred.Σ
    @assert P_p isa PSDMatrix || P_p isa Matrix
    if (P_p isa PSDMatrix && iszero(P_p.R)) || (P_p isa Matrix && iszero(P_p))
        copy!(x_out, x_pred)
        return x_out
    end

    D = length(m_p)

    # K = P_p * H' / S
    _S = if S isa PSDMatrix
        _matmul!(C_dxd, S.R', S.R)
    else
        copy!(C_dxd, S)
    end

    K = if P_p isa PSDMatrix
        _matmul!(K1_cache, P_p.R, H')
        _matmul!(K2_cache, P_p.R', K1_cache)
    else
        _matmul!(K2_cache, P_p, H')
    end

    S_chol = try
        cholesky!(_S)
    catch e
        if !(e isa PosDefException)
            rethrow(e)
        end
        @warn "Can't compute the update step with cholesky; using qr instead"
        @assert S isa PSDMatrix
        Cholesky(qr(S.R).R, :U, 0)
    end
    rdiv!(K, S_chol)

    loglikelihood = zero(eltype(K))
    loglikelihood = pn_logpdf!(measurement, S_chol, copy(z))

    # x_out.μ .= m_p .+ K * (0 .- z)
    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)

    # M_cache .= I(D) .- mul!(M_cache, K, H)
    _matmul!(M_cache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        M_cache[i, i] += 1
    end

    fast_X_A_Xt!(x_out.Σ, P_p, M_cache)

    if !iszero(R)
        out_Sigma_R = [x_out.Σ.R; sqrt.(R) * K']
        x_out.Σ.R .= triangularize!(out_Sigma_R; cachemat=M_cache)
    end

    return x_out, loglikelihood
end
function compute_nll_and_update!(x, u, H, R, m_tmp, ZERO_DATA, cache)
    msmnt = measure!(x, H, R, m_tmp)
    msmnt.μ .-= u
    # nll = -pn_logpdf!(msmnt, cholesky(msmnt.Σ), copy(msmnt.μ))
    # copy!(x, ProbNumDiffEq.update(x, msmnt, H))

    @unpack K1, x_tmp2, m_tmp = cache
    d = length(u)
    # KC, MC, SC = view(K1, :, 1:d), x_tmp2.Σ.mat, view(m_tmp.Σ.mat, 1:d, 1:d)
    xout = cache.x_tmp
    # ProbNumDiffEq.update!(xout, x, msmnt, H, KC, MC, SC)

    @unpack x_tmp2, m_tmp, C_DxD = cache
    C_dxd = view(cache.C_dxd, 1:d, 1:d)
    K1 = view(cache.K1, :, 1:d)
    K2 = view(cache.C_Dxd, :, 1:d)
    _, ll = update!(xout, x, msmnt, R, H, K1, K2, C_DxD, C_dxd)
    nll = -ll

    copy!(x, xout)
    return nll
end

function project_to_solution_space!(u_probs, states, projmat)
    for (pu, x) in zip(u_probs, states)
        _gaussian_mul!(pu, projmat, x)
    end
    return u_probs
end
