mutable struct DataUpdateLogLikelihood{T<:Number}
    ll::T
end

using LinearAlgebra: NumberArray
function DataUpdateCallback(
    data::NamedTuple{(:t, :u)};
    observation_matrix=I,
    observation_noise_cov,
    loglikelihood::Union{DataUpdateLogLikelihood,Nothing}=nothing,
)
    function affect!(integ)
        times, values = data.t, data.u
        idx = findfirst(isequal(integ.t), times)
        val = values[idx]

        o = length(val)

        @unpack K1, C_DxD, E0, C_dxd, C_Dxd, C_d = integ.cache
        K1 = view(K1, :, 1:o)
        C_dxd = view(C_dxd, 1:o, 1:o)
        C_Dxd = view(C_Dxd, :, 1:o)
        C_d = view(C_d, 1:o)

        x = integ.cache.x

        H = observation_matrix * E0

        obs_mean = H * x.μ - val

        R = if observation_noise_cov isa Number
            observation_noise_cov * Eye(o)
        else
            observation_noise_cov
        end

        _A = x.Σ.R * H'
        # obs_cov = PSDMatrix(qr!([x.Σ.R * H'; sqrt(observation_noise_cov) * Eye(o)]).R)
        obs_cov = _A'_A + R
        obs = Gaussian(obs_mean, obs_cov)

        _x = copy!(integ.cache.x_tmp, x)
        _, ll = update!(x, _x, obs, H, K1, C_Dxd, C_DxD, C_dxd, C_d)
        if !isnothing(loglikelihood)
            loglikelihood.ll += ll
        end
    end
    return PresetTimeCallback(data.t, affect!; save_positions=(false, false))
end
