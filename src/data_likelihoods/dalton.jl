"""
$(TYPEDSIGNATURES)

Compute the DALTON [wu23dalton](@cite) approximate negative log-likelihood (NLL) of the data.

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
* [wu23dalton](@cite) Wu et al, "Data-Adaptive Probabilistic Likelihood Approximation for Ordinary Differential Equations", arXiv (2023)
"""
function _dalton_data_loglik(
    prob::SciMLBase.AbstractODEProblem,
    alg::AbstractEK,
    args...;
    # observation model
    observation_matrix=I,
    observation_noise_cov::Union{Number, AbstractMatrix},
    # data
    data::NamedTuple{(:t, :u)},
    kwargs...
)

    if alg.smooth
        str = "The passed algorithm performs smoothing, but `dalton_nll` can be used without. " *
            "You might want to set `smooth=false` to imprpove performance."
        @warn str
    end
    if !(:adaptive in keys(kwargs))
        throw(ArgumentError("`dalton_nll` only works with fixed step sizes. Set `adaptive=false`."))
    end

    if :tstops in keys(kwargs)
        str = "The passed `tstops` argument will be extended with the observation locations `data.t`."
        @warn str
    end
    tstops = union(data.t, get(kwargs, :tstops, []))

    data_ll = DataUpdateLogLikelihood(zero(eltype(prob.p)))

    cb = DataUpdateCallback(
        data; observation_matrix, observation_noise_cov,
        loglikelihood=data_ll)

    sol_with_data = solve(
        prob, alg, args...;
        callback=cb,
        save_everystep=false,
        kwargs...,
        tstops,
    )

    sol_without_data = solve(
        prob, alg, args...;
        save_everystep=false,
        kwargs...,
        tstops,
    )

    dalton_ll = (data_ll.ll
                 + sol_with_data.pnstats.log_likelihood
                 - sol_without_data.pnstats.log_likelihood)

    return dalton_ll
end

