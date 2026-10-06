"""
$(TYPEDEF)

This is just a container for the data log-likelihood such that it can be computed in the
[`DataUpdateCallback`](@ref) and is accessible on the outside, by mutating the `ll` field.
"""
mutable struct DataUpdateLogLikelihood{T<:Number}
    ll::T
end

@doc raw"""
    DataUpdateCallback(
        data::NamedTuple{(:t, :u)};
        observation_matrix=I,
        observation_noise_cov,
        loglikelihood::Union{DataUpdateLogLikelihood,Nothing}=nothing,
        save_positions=(false, false),
        kwargs...
    )

Update the state accoding to the linear observations during the filter pass.

`DataUpdateCallback` returns a `DiffEqCallbacks.PresetTimeCallback`, which (i) adjusts the
`tstops` to include the observation times (`data.t`) and (ii) whenever a time step
coincides with the data locations, it updates the state on the data point according to
the observation model
```math
\begin{aligned}
y(t) &= H x(t) + \varepsilon(t), \quad \varepsilon(t) \sim \mathcal{N}(0, R),
\end{aligned}
```
where ``H`` is the observation matrix (`observation_matrix`) and
``R`` is the observation noise covariance (`observation_noise_cov`).
For second-order ODEs, the observation matrix acts on `u` only, not on `du`.

The rows of the observation matrix must not be zero. Observation matrices other than `I`
(or a multiple of it), e.g. for partial observations (`o < d`), work with all solvers
except the `EK0` with an `IWP` prior and a scalar diffusion, as by default; there, the
observation noise must also be isotropic: a scalar variance, a `UniformScaling` or an
`Eye`. With a block-diagonal covariance (the `DiagonalEK1`, or the `EK0` with a
multivariate diffusion), observation matrices must select dimensions, i.e. each row must
have exactly one nonzero entry (any scaling is fine) and no dimension may be observed
twice, and the observation noise must be a scalar variance, a `UniformScaling` or a
`Diagonal`.

By passing a [`DataUpdateLogLikelihood`](@ref) object with the `loglikelihood` keyword
argument, the log-likelihood of the data is computed and stored in the `ll` field, and can
be accessed after call to `solve`.
"""
function DataUpdateCallback(
    data::NamedTuple{(:t, :u)};
    observation_matrix=I,
    observation_noise_cov,
    loglikelihood::Union{DataUpdateLogLikelihood,Nothing}=nothing,
    save_positions=(false, false),
    kwargs...,
)
    check_observation_noise_cov(observation_noise_cov)
    function affect!(integ)
        times, values = data.t, data.u
        idx = findfirst(isequal(integ.t), times)
        val = values[idx]
        o = length(val)

        H, R = make_observation_model(
            integ.cache, observation_matrix, observation_noise_cov; o)

        # The initial value is known exactly, so no update is needed, only the likelihood
        if integ.iter == 0
            ll = initial_data_loglik(integ.u, val, observation_matrix, R)
            isnothing(loglikelihood) || (loglikelihood.ll += ll)
            return nothing
        end

        ll = update_on_data!(integ.cache.x, H, val, R; cache=integ.cache).loglikelihood

        if !isnothing(loglikelihood)
            loglikelihood.ll += ll
        end
    end
    return PresetTimeCallback(data.t, affect!; save_positions, kwargs...)
end

function initial_data_loglik(u0, val, M, R::PSDMatrix)
    (u0 isa RecursiveArrayTools.ArrayPartition) && (u0 = u0.x[2]) # for 2ndOrderODEs
    return logpdf(Gaussian(M * vec(u0), Matrix(R)), val)
end
