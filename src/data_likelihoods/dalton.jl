"""
$(TYPEDSIGNATURES)

Compute the DALTON [Wu & Lysy (2023)](@cite wu23dalton) approximate negative log-likelihood (NLL) of the data.

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
* [Wu & Lysy (2023)](@cite wu23dalton) "Data-Adaptive Probabilistic Likelihood Approximation for Ordinary Differential Equations"
"""
function dalton_data_loglik(
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
    if alg.smooth
        str =
            "The passed algorithm performs smoothing, but `dalton_data_loglik` can be used " *
            "without. You might want to set `smooth=false` to improve performance."
        @warn str
    end
    if !(:adaptive in keys(kwargs))
        str = "`dalton_data_loglik` only works with fixed step sizes. Set `adaptive=false`."
        throw(ArgumentError(str))
    end

    tstops = union(data.t, get(kwargs, :tstops, []))

    data_ll = DataUpdateLogLikelihood{Real}(0)

    cb = DataUpdateCallback(
        data; observation_matrix, observation_noise_cov,
        loglikelihood=data_ll)

    sol_with_data = solve(
        prob, alg, args...;
        save_everystep=false,
        kwargs...,
        tstops,
        callback=CallbackSet(cb, get(kwargs, :callback, nothing)),
    )

    sol_without_data = solve(
        prob, alg, args...;
        save_everystep=false,
        kwargs...,
        tstops,
    )

    sol_with_data_pn_ll = sol_with_data.pnstats.log_likelihood
    sol_without_data_pn_ll = sol_without_data.pnstats.log_likelihood
    dalton_ll = data_ll.ll + sol_with_data_pn_ll - sol_without_data_pn_ll
    return dalton_ll
end
