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

