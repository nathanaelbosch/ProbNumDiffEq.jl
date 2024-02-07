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

        x = integ.cache.x
        H = observation_matrix * integ.cache.E0
        obs_mean = H * x.μ - val

        R = if observation_noise_cov isa PSDMatrix
            observation_noise_cov
        elseif observation_noise_cov isa Number
            PSDMatrix(sqrt(observation_noise_cov) * Eye(o))
        elseif observation_noise_cov isa UniformScaling
            PSDMatrix(sqrt(observation_noise_cov.λ) * Eye(o))
        else
            PSDMatrix(cholesky(observation_noise_cov).U)
        end

        # _A = x.Σ.R * H'
        # obs_cov = _A'_A + R
        obs_cov = PSDMatrix(qr!([x.Σ.R * H'; R.R]).R)
        obs = Gaussian(obs_mean, obs_cov)

        @unpack x_tmp, K1, C_DxD, C_dxd, C_Dxd, C_d = integ.cache
        K1 = view(K1, :, 1:o)
        C_dxd = view(C_dxd, 1:o, 1:o)
        C_Dxd = view(C_Dxd, :, 1:o)
        C_d = view(C_d, 1:o)
        _x = copy!(x_tmp, x)
        _, ll = update!(x, _x, obs, H, K1, C_Dxd, C_DxD, C_dxd, C_d; R=R)

        if !isnothing(loglikelihood)
            loglikelihood.ll += ll
        end
    end
    return PresetTimeCallback(data.t, affect!; save_positions=(false, false))
end
