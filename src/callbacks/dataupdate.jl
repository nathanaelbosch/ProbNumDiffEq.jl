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

        @unpack x, E0, m_tmp = integ.cache
        M = observation_matrix

        if o != d && integ.cache.covariance_factorization isa BlockDiagonalCovariance
            obs_indices = _partial_obs_indices(M, d)
            _x = copy!(integ.cache.x_tmp, x)
            ll = _partial_block_update!(
                x, _x, obs_indices, val, E0, observation_noise_cov)
        elseif o != d && !(integ.cache.covariance_factorization isa DenseCovariance)
            error("Partial observations only work with the EK1 and DiagonalEK1 right now")
        else
            H = M * E0

            obs_mean = _matmul!(view(m_tmp.μ, 1:o), H, x.μ)
            obs_mean .-= val

            R = cov2psdmatrix(observation_noise_cov; d=o)
            R = to_factorized_matrix(integ.cache.covariance_factorization, R)

            obs_cov = PSDMatrix(make_obscov_sqrt(x.Σ.R, H, R.R))
            obs = Gaussian(obs_mean, obs_cov)

            _cache = make_obssized_cache(integ.cache; o)
            @unpack K1, C_DxD, C_dxd, C_Dxd, C_d = _cache
            _x = copy!(integ.cache.x_tmp, x)
            _, ll = update!(x, _x, obs, H, K1, C_Dxd, C_DxD, C_dxd, C_d; R=R)
        end

        if !isnothing(loglikelihood)
            loglikelihood.ll += ll
        end
    end
    return PresetTimeCallback(data.t, affect!; save_positions, kwargs...)
end

function _partial_obs_indices(M::AbstractMatrix, d::Int)
    o = size(M, 1)
    @assert size(M, 2) == d
    indices = Vector{Int}(undef, o)
    for k in 1:o
        row = view(M, k, :)
        nz = findall(!iszero, row)
        if length(nz) != 1 || row[nz[1]] != 1
            throw(
                ArgumentError(
                    "Partial observations with `DiagonalEK1` require a dimension-selection " *
                    "observation matrix (each row must select exactly one dimension). " *
                    "Got a row that mixes dimensions or has a non-unit scaling."),
            )
        end
        indices[k] = nz[1]
    end
    if length(unique(indices)) != o
        throw(
            ArgumentError(
                "Partial observations with `DiagonalEK1` require each observed dimension " *
                "to appear exactly once. Got repeated dimensions."),
        )
    end
    return indices
end

make_obscov_sqrt(PR::AbstractMatrix, H::AbstractMatrix, RR::AbstractMatrix) =
    qr!([PR * H'; RR]).R
make_obscov_sqrt(
    PR::IsometricKroneckerProduct,
    H::IsometricKroneckerProduct,
    RR::IsometricKroneckerProduct,
) =
    IsometricKroneckerProduct(PR.rdim, make_obscov_sqrt(PR.B, H.B, RR.B))
make_obscov_sqrt(PR::BlocksOfDiagonals, H::BlocksOfDiagonals, RR::BlocksOfDiagonals) =
    BlocksOfDiagonals([
        make_obscov_sqrt(blocks(PR)[i], blocks(H)[i], blocks(RR)[i]) for
        i in eachindex(blocks(PR))
    ])

function _partial_block_update!(
    x_out::SRGaussian{T,<:BlocksOfDiagonals},
    x_pred::SRGaussian{T,<:BlocksOfDiagonals},
    obs_indices::Vector{Int},
    val::AbstractVector,
    proj::BlocksOfDiagonals,
    observation_noise_cov;
) where {T}
    d = length(blocks(x_out.Σ.R))
    q1 = size(blocks(x_out.Σ.R)[1], 1)
    obs_per_dim = size(blocks(proj)[1], 1)

    copy!(x_out, x_pred)

    K1_cache = Matrix{T}(undef, q1, obs_per_dim)
    K2_cache = Matrix{T}(undef, q1, obs_per_dim)
    M_cache = Matrix{T}(undef, q1, q1)
    C_dxd = Matrix{T}(undef, obs_per_dim, obs_per_dim)
    C_d = Vector{T}(undef, obs_per_dim)
    z_k = Vector{T}(undef, obs_per_dim)
    RR_k = zeros(T, obs_per_dim, obs_per_dim)
    R_k = PSDMatrix(RR_k)

    ll = zero(T)
    for (k, i) in enumerate(obs_indices)
        H_k = blocks(proj)[i]
        x_out_k = Gaussian(
            view(x_out.μ, i:d:length(x_out.μ)),
            PSDMatrix(x_out.Σ.R.blocks[i]))
        x_pred_k = Gaussian(
            view(x_pred.μ, i:d:length(x_pred.μ)),
            PSDMatrix(x_pred.Σ.R.blocks[i]))

        mul!(z_k, H_k, view(x_pred.μ, i:d:length(x_pred.μ)))
        val_range = ((k-1)*obs_per_dim+1):(k*obs_per_dim)
        z_k .-= view(val, val_range)

        r_k = _get_obs_noise_var(observation_noise_cov, k)
        sqrt_r_k = sqrt(r_k)
        fill!(RR_k, zero(T))
        for j in 1:obs_per_dim
            RR_k[j, j] = sqrt_r_k
        end

        obs_cov_sqrt_k = make_obscov_sqrt(x_pred.Σ.R.blocks[i], H_k, RR_k)
        obs_k = Gaussian(z_k, PSDMatrix(obs_cov_sqrt_k))

        _, _ll = update!(
            x_out_k, x_pred_k, obs_k, H_k,
            K1_cache, K2_cache, M_cache, C_dxd, C_d; R=R_k)
        ll += _ll
    end
    return ll
end

_get_obs_noise_var(cov::Number, k::Int) = cov
_get_obs_noise_var(cov::UniformScaling, k::Int) = cov.λ
_get_obs_noise_var(cov::Diagonal, k::Int) = cov.diag[k]

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
