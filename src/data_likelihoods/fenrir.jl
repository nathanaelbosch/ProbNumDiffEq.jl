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
    observation_noise_cov::Union{Number,UniformScaling,AbstractMatrix},
    # data
    data::NamedTuple{(:t, :u)},
    kwargs...,
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
    o = length(data.u[1])
    R = cov2psdmatrix(observation_noise_cov; d=o)
    LL, _, _ = fit_pnsolution_to_data!(sol, R, data; proj=observation_matrix)

    return LL
end

function fit_pnsolution_to_data!(
    sol::AbstractProbODESolution,
    observation_noise_cov::PSDMatrix,
    data::NamedTuple{(:t, :u)};
    proj=I,
)
    @unpack cache, backward_kernels = sol
    @unpack A, Q, x_tmp, x_tmp2, m_tmp, C_DxD, C_3DxD = cache

    LL = zero(eltype(sol.prob.p))

    o = length(data.u[1])
    @unpack x_tmp, C_dxd, C_d, K1, C_Dxd, C_DxD, m_tmp = cache
    _cache = (
        x_tmp=x_tmp,
        C_DxD=C_DxD,
        C_Dxd=view(C_Dxd, :, 1:o),
        C_dxd=view(C_dxd, 1:o, 1:o),
        C_d=view(C_d, 1:o),
        K1=view(K1, :, 1:o),
        K2=view(C_Dxd, :, 1:o),
        m_tmp=Gaussian(view(m_tmp.μ, 1:o), PSDMatrix(view(m_tmp.Σ.R, :, 1:o))),
    )

    x_posterior = copy(sol.x_filt) # the object to be filled
    state2data_projmat = proj * cache.SolProj

    # First update on the last data point
    if sol.t[end] in data.t
        _, ll = measure_and_update!(
            x_posterior[end],
            data.u[end],
            state2data_projmat,
            observation_noise_cov,
            _cache,
        )
        LL += ll
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
            _, ll = measure_and_update!(
                x_posterior[i],
                data.u[data_idx],
                state2data_projmat,
                observation_noise_cov,
                _cache,
            )
            if !isinf(ll)
                LL += ll
            end
            data_idx -= 1
        end
    end
    @assert data_idx == 0 # to make sure we went through all the data

    return LL, sol.t, x_posterior
end

function measure_and_update!(x, u, H, R::PSDMatrix, cache)
    z, S = cache.m_tmp
    _matmul!(z, H, x.μ)
    z .-= u
    fast_X_A_Xt!(S, x.Σ, H)
    # _S = PSDMatrix(S.R'S.R + R.R'R.R)
    _S = PSDMatrix(triangularize!([S.R; R.R], cachemat=cache.C_DxD))
    msmnt = Gaussian(z, _S)

    return update!(x, copy!(cache.x_tmp, x), msmnt, H; R=R, cache)
end
