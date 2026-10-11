function filtering_data_loglik(
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
    if alg.smooth
        str =
            "The passed algorithm performs smoothing, but `filtering_data_loglik` can be used " *
            "without. You might want to set `smooth=false` to improve performance."
        @warn str
    end

    data_ll = DataUpdateLogLikelihood{Real}(0.0)

    cb = DataUpdateCallback(
        data; observation_matrix, observation_noise_cov,
        loglikelihood=data_ll)

    solve(
        prob, alg, args...;
        save_everystep=false,
        kwargs...,
        callback=CallbackSet(cb, get(kwargs, :callback, nothing)),
    )

    return data_ll.ll
end
