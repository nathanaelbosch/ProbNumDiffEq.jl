mutable struct DataUpdateLogLikelihood{T<:Number}
    ll::T
end

function DataUpdateCallback(
    data::NamedTuple{(:t, :u)};
    observation_noise_variance::Number,
    loglikelihood::Union{DataUpdateLogLikelihood,Nothing}=nothing
)
    function affect!(integ)
        times, values = data.t, data.u
        idx = findfirst(isequal(integ.t), times)
        val = values[idx]

        d = length(val)
        @assert d == length(integ.u)

        @unpack K1, C_DxD, E0, C_dxd, C_Dxd, C_d = integ.cache

        x = integ.cache.x

        obs_mean = E0 * x.μ - val
        obs_cov = PSDMatrix(
            qr!([x.Σ.R * E0'; sqrt(observation_noise_variance) * Eye(d)]).R
        )
        obs = Gaussian(obs_mean, obs_cov)

        _x = copy!(integ.cache.x_tmp, x)
        _, ll = update!(x, _x, obs, E0, K1, C_Dxd, C_DxD, C_dxd, C_d)
        if !isnothing(loglikelihood)
            loglikelihood.ll += ll
        end
    end
    return PresetTimeCallback(data.t, affect!; save_positions=(false, false))
end
