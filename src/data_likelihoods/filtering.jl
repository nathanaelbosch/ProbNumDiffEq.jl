function filtering_data_loglik(
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
    if :tstops in keys(kwargs)
        str = "The passed `tstops` argument will be extended with the observation locations `data.t`."
        @warn str
    end
    tstops = union(data.t, get(kwargs, :tstops, []))

    data_ll = DataUpdateLogLikelihood(zero(eltype(prob.p)))

    cb = DataUpdateCallback(
        data; observation_matrix, observation_noise_cov,
        loglikelihood=data_ll)

    solve(
        prob, alg, args...;
        callback=cb,
        save_everystep=false,
        tstops,
        kwargs...,
    )

    return data_ll.ll
end
