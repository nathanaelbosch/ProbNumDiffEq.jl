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

Partial observations (`o < d`) work with all solvers except the `EK0` with an `IWP` prior
and a scalar diffusion, as by default. With a block-diagonal covariance (the `DiagonalEK1`,
or the `EK0` with a multivariate diffusion), observation matrices must select dimensions,
i.e. each row must have exactly one nonzero entry (any scaling is fine), and the
observation noise must be uncorrelated.

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

        H, R = observation_model(integ.cache, observation_matrix, observation_noise_cov; o)

        # The initial value is known exactly, so no update is needed, only the likelihood
        if integ.iter == 0
            ll = initial_data_loglik(integ.u, val, observation_matrix, R)
            isnothing(loglikelihood) || (loglikelihood.ll += ll)
            return nothing
        end

        _, ll = measure_and_update!(
            integ.cache.x, val, H, R, make_obssized_cache(integ.cache; o))

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

function check_observation_noise_cov(cov)
    if !_is_positive(cov)
        throw(ArgumentError("The observation noise covariance must be positive definite."))
    end
end
_is_positive(cov::Number) = cov > 0
_is_positive(cov::UniformScaling) = cov.λ > 0
_is_positive(cov::Diagonal) = all(>(0), cov.diag)
_is_positive(cov::AbstractMatrix) = isposdef(Matrix(cov))

function observation_model(cache, M, noise_cov; o)
    d = cache.d
    rows, cols = M isa UniformScaling ? (d, d) : size(M)
    if cols != d
        throw(
            ArgumentError(
                "The observation matrix must have one column per ODE dimension " *
                "(d = $d), but has size $(size(M)). For second-order ODEs, it acts on " *
                "`u` only."),
        )
    end
    if rows != o
        throw(
            DimensionMismatch(
                "The observation matrix maps the state to $rows values, but the data " *
                "has $o entries."),
        )
    end
    if o > d
        throw(
            ArgumentError(
                "The observation matrix has $o rows, but at most d = $d (the ODE " *
                "dimension) are supported right now."),
        )
    end
    fac = cache.covariance_factorization
    if fac isa BlockDiagonalCovariance
        return block_selection_model(fac, M, cache.E0, noise_cov; o)
    end
    if fac isa IsometricKroneckerCovariance && o != d
        throw(
            ArgumentError(
                "Partial observations require a `DenseCovariance` or " *
                "`BlockDiagonalCovariance` covariance structure (like the `EK1` or " *
                "`DiagonalEK1`); they are not supported with the isometric-kronecker " *
                "structure right now"),
        )
    end
    R = to_factorized_matrix(fac, cov2psdmatrix(noise_cov; d=o))
    return M * cache.E0, R
end
function block_selection_model(
    fac::BlockDiagonalCovariance{T}, M, E0, noise_cov; o,
) where {T}
    obs_dims = _selected_dims(M, o)
    H = BlocksOfDiagonals([M[k, i] * blocks(E0)[i] for (k, i) in enumerate(obs_dims)])
    R = to_factorized_matrix(
        BlockDiagonalCovariance{T}(o, fac.q),
        cov2psdmatrix(_diagonal_noise(noise_cov); d=o))
    return BlockSelection(H, obs_dims), R
end

_diagonal_noise(cov::Union{Number,UniformScaling,Diagonal}) = cov
function _diagonal_noise(cov::AbstractMatrix)
    C = Matrix(cov)
    if !isdiag(C)
        throw(
            ArgumentError(
                "Observations with a block-diagonal covariance structure require " *
                "uncorrelated observation noise, i.e. a diagonal noise covariance."),
        )
    end
    return Diagonal(diag(C))
end

_selected_dims(::UniformScaling, o) = collect(1:o)
_selected_dims(::Diagonal, o) = collect(1:o)
function _selected_dims(M::AbstractMatrix, o)
    indices = Vector{Int}(undef, o)
    for k in 1:o
        row = view(M, k, :)
        nz = findall(!iszero, row)
        if length(nz) != 1
            problem = isempty(nz) ? "has no nonzero entry" : "mixes dimensions"
            throw(
                ArgumentError(
                    "Observation matrices with a block-diagonal covariance structure " *
                    "(e.g. the `DiagonalEK1`) must select dimensions: each row must " *
                    "have exactly one nonzero entry (any scaling is fine). " *
                    "Row $k $problem."),
            )
        end
        indices[k] = nz[1]
    end
    if !allunique(indices)
        throw(
            ArgumentError(
                "Observation matrices with a block-diagonal covariance structure " *
                "(e.g. the `DiagonalEK1`) must observe each dimension at most once. " *
                "Got repeated dimensions."),
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

# Partial observation matrix for block-diagonal covariances: block `k` of `H` observes
# block `obs_dims[k]` of the state, and the other blocks of the state are not observed
struct BlockSelection{HT<:BlocksOfDiagonals}
    H::HT
    obs_dims::Vector{Int}
end
function _matmul!(z::AbstractVector, H::BlockSelection, μ::AbstractVector)
    o, d = length(H.obs_dims), length(μ) ÷ size(blocks(H.H)[1], 2)
    @views for (k, i) in enumerate(H.obs_dims)
        _matmul!(z[k:o:end], blocks(H.H)[k], μ[i:d:end])
    end
    return z
end
make_obscov_sqrt(PR::BlocksOfDiagonals, H::BlockSelection, RR::BlocksOfDiagonals) =
    make_obscov_sqrt(BlocksOfDiagonals(blocks(PR)[H.obs_dims]), H.H, RR)
update!(x_out, x_pred, measurement, H::BlockSelection, K1, K2, M, C_dxd, C_d; R=nothing) =
    update!(x_out, x_pred, measurement, H.H, K1, K2, M, C_dxd, C_d; R, H.obs_dims)

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
function make_obssized_cache(::BlockDiagonalCovariance, cache; o)
    # The block-wise `update!` only uses the first `o` blocks of the caches
    @unpack K1, C_DxD, C_dxd, C_Dxd, C_d, m_tmp, x_tmp = cache
    return (
        K1=K1,
        C_dxd=C_dxd,
        C_Dxd=C_Dxd,
        C_d=C_d,
        C_DxD=C_DxD,
        m_tmp=Gaussian(view(m_tmp.μ, 1:o), m_tmp.Σ),
        x_tmp=x_tmp,
    )
end
