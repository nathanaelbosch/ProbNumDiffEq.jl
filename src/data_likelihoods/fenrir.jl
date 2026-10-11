"""
$(TYPEDSIGNATURES)

Compute the Fenrir [Tronarp et al. (2022)](@cite tronarp22fenrir) approximate negative log-likelihood (NLL) of the data.

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
- `alg::ODEFilter`: the probabilistic ODE solver to be used; use `EK1` for best results.
- `data::NamedTuple{(:t, :u)}`: the data to be fitted
- `observation_matrix::Union{AbstractMatrix,UniformScaling}`:
  the matrix which maps the ODE state to the measurements; typically a projection matrix.
  For second-order ODEs, it acts on `u` only, not on `du`. Its rows must not be zero.
  Observation matrices other than `I` (or a multiple of it), e.g. for partial
  observations (`o < d`), work with all solvers except the `EK0` with an `IWP` prior and
  a scalar diffusion, as by default. With a block-diagonal covariance (the `DiagonalEK1`,
  or the `EK0` with a multivariate diffusion), observation matrices must select
  dimensions, i.e. each row must have exactly one nonzero entry (any scaling is fine) and
  no dimension may be observed twice.
- `observation_noise_cov::Union{Number,UniformScaling,AbstractMatrix}`: the observation
  noise covariance, or a scalar variance. With a block-diagonal covariance it must be a
  scalar, a `UniformScaling` or a `Diagonal`, and with the `EK0` with an `IWP` prior and
  a scalar diffusion a scalar, a `UniformScaling` or an `Eye`.

# Reference
* [Tronarp et al. (2022)](@cite tronarp22fenrir) "Fenrir: Physics-Enhanced Regression for Initial Value Problems", ICML
"""
function fenrir_data_loglik(
    prob::SciMLBase.AbstractODEProblem,
    alg::ODEFilter,
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
    check_observation_noise_cov(observation_noise_cov)
    tstops = union(data.t, get(kwargs, :tstops, []))

    integ = init(prob, alg, args...; kwargs..., tstops)

    # Build the observation model before the solve, such that unsupported inputs fail early
    H, R = make_observation_model(
        integ.cache, observation_matrix, observation_noise_cov; o=length(data.u[1]))

    T = prob.tspan[2] - prob.tspan[1]
    step!(integ, T, false) # basically `solve!` but this prevents smoothing
    sol = integ.sol
    if sol.retcode !== SciMLBase.ReturnCode.Success &&
       sol.retcode !== SciMLBase.ReturnCode.Default
        @error "The PN ODE solver did not succeed!" sol.retcode
        return -Inf * one(eltype(integ.u))
    end

    # Fit the ODE solution / PN posterior to the provided data; this is the actual Fenrir
    LL, _, _ = fit_pnsolution_to_data!(sol, H, R, data)

    return LL
end

function fit_pnsolution_to_data!(
    sol::AbstractProbODESolution,
    H,
    observation_noise_cov::PSDMatrix,
    data::NamedTuple{(:t, :u)},
)
    @unpack cache, backward_kernels = sol
    @unpack C_DxD, C_3DxD = cache

    LL = zero(eltype(sol.x_filt[1].μ))

    _cache = make_obssized_cache(cache; o=length(data.u[1]))

    x_posterior = copy(sol.x_filt) # the object to be filled

    # First update on the last data point, if it lies at the end of the solution
    data_idx = length(data.u)
    if sol.t[end] == data.t[data_idx]
        _, ll = measure_and_update!(
            x_posterior[end],
            data.u[data_idx],
            H,
            observation_noise_cov,
            _cache,
        )
        LL += ll
        data_idx -= 1
    end

    # Now iterate backwards
    for i in (length(x_posterior)-1):-1:1
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
                H,
                observation_noise_cov,
                _cache,
            )
            LL += ll
            data_idx -= 1
        end
    end
    @assert data_idx == 0 # to make sure we went through all the data

    return LL, sol.t, x_posterior
end

function measure_and_update!(x, u, H, R::PSDMatrix, cache)
    z = view(mean(cache.m_tmp), 1:length(u))
    _matmul!(z, H, x.μ)
    z .-= u
    S = PSDMatrix(make_obscov_sqrt(x.Σ.R, H, R.R))
    msmnt = Gaussian(z, S)

    return update!(x, copy!(cache.x_tmp, x), msmnt, H; R=R, cache)
end
