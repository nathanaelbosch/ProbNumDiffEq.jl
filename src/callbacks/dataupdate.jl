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
    function affect!(integ)
        times, values = data.t, data.u
        idx = findfirst(isequal(integ.t), times)
        val = values[idx]

        o = length(val)
        d = integ.cache.d

        @unpack x, E0, m_tmp, G1 = integ.cache
        M = observation_matrix
        H = M * E0

        obs_mean = _matmul!(view(m_tmp.μ, 1:o), H, x.μ)
        obs_mean .-= val

        R = cov2psdmatrix(observation_noise_cov; d=o)
        R = to_factorized_matrix(integ.cache.covariance_factorization, R)

        # _A = x.Σ.R * H'
        # obs_cov = _A'_A + R
        obs_cov = PSDMatrix(make_obscov_sqrt(x.Σ.R, H, R.R))

        obs = Gaussian(obs_mean, obs_cov)

        if o != d && !(integ.alg isa EK1)
            error("Partial observations only work with the EK1 right now")
        end
        _cache = make_obssized_cache(integ.cache; o)
        @unpack K1, C_DxD, C_dxd, C_Dxd, C_d = _cache
        _x = copy!(integ.cache.x_tmp, x)
        _, ll = update!(x, _x, obs, H, K1, C_Dxd, C_DxD, C_dxd, C_d; R=R)

        if !isnothing(loglikelihood)
            loglikelihood.ll += ll
        end
    end
    return PresetTimeCallback(data.t, affect!; save_positions, kwargs...)
end

make_obscov_sqrt(PR::AbstractMatrix, H::AbstractMatrix, RR::AbstractMatrix) =
    qr!([PR * H'; RR]).R
make_obscov_sqrt(
    PR::IsometricKroneckerProduct,
    H::IsometricKroneckerProduct,
    RR::IsometricKroneckerProduct,
) =
    IsometricKroneckerProduct(PR.ldim, make_obscov_sqrt(PR.B, H.B, RR.B))
make_obscov_sqrt(PR::BlockDiag, H::BlockDiag, RR::BlockDiag) =
    BlockDiag([
        make_obscov_sqrt(blocks(PR)[i], blocks(H)[i], blocks(RR)[i]) for
        i in eachindex(blocks(PR))
    ])

function make_obssized_cache(cache; o)
    if o == cache.d
        return cache
    else
        return make_obssized_cache(cache.covariance_factorization, cache; o)
    end
end
function make_obssized_cache(::DenseCovariance, cache; o)
    @unpack K1, C_DxD, C_dxd, C_Dxd, C_d, m_tmp, x_tmp = cache
    return (
        K1=view(K1, :, 1:o),
        C_dxd=view(C_dxd, 1:o, 1:o),
        C_Dxd=view(C_Dxd, :, 1:o),
        C_d=view(C_d, 1:o),
        C_DxD=C_DxD,
        m_tmp=Gaussian(view(m_tmp.μ, 1:o), view(m_tmp.Σ, 1:o, 1:o)),
        x_tmp=x_tmp,
    )
end
