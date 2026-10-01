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
observation noise must also be a multiple of the identity. With a block-diagonal
covariance (the `DiagonalEK1`, or the `EK0` with a multivariate diffusion), observation
matrices must select dimensions, i.e. each row must have exactly one nonzero entry (any
scaling is fine), and the observation noise must be uncorrelated.

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
    _is_positive(cov) || nonpositive_noise_error()
    return nothing
end
_is_positive(cov::Number) = cov > 0
_is_positive(cov::UniformScaling) = cov.λ > 0
_is_positive(cov::Diagonal) = all(>(0), cov.diag)
_is_positive(cov::AbstractMatrix) = isposdef(Matrix(cov))

function observation_model(cache, M, noise_cov; o)
    d = cache.d
    M isa Number && (M = M * I)
    rows, cols = M isa UniformScaling ? (d, d) : size(M)
    cols == d || column_count_error(M, d)
    rows == o || row_count_error(rows, o)
    o <= d || too_many_rows_error(o, d)
    k = _zero_row(M)
    isnothing(k) || zero_row_error(k)
    noise_cov isa AbstractMatrix && size(noise_cov) != (o, o) &&
        noise_size_error(noise_cov, o)
    return observation_model(cache.covariance_factorization, M, cache.E0, noise_cov; o)
end
function observation_model(fac::DenseCovariance, M, E0, noise_cov; o)
    return M * E0, to_factorized_matrix(fac, cov2psdmatrix(noise_cov; d=o))
end
function observation_model(fac::IsometricKroneckerCovariance, M, E0, noise_cov; o)
    M isa UniformScaling || kronecker_matrix_error()
    R = to_factorized_matrix(fac, cov2psdmatrix(_isotropic_noise(noise_cov); d=o))
    return M * E0, R
end
function observation_model(fac::BlockDiagonalCovariance{T}, M, E0, noise_cov; o) where {T}
    obs_dims = _selected_dims(M, o)
    H = BlocksOfDiagonals([M[k, i] * blocks(E0)[i] for (k, i) in enumerate(obs_dims)])
    R = to_factorized_matrix(
        BlockDiagonalCovariance{T}(o, fac.q),
        cov2psdmatrix(_diagonal_noise(noise_cov); d=o))
    return BlockSelection(H, obs_dims), R
end

_zero_row(M::UniformScaling) = iszero(M.λ) ? 1 : nothing
_zero_row(M::Diagonal) = findfirst(iszero, M.diag)
_zero_row(M::AbstractMatrix) = findfirst(row -> all(iszero, row), eachrow(M))

_isotropic_noise(cov::Union{Number,UniformScaling}) = cov
function _isotropic_noise(cov::AbstractMatrix)
    C = Matrix(cov)
    isdiag(C) && allequal(diag(C)) || anisotropic_noise_error()
    return C[1, 1]
end

_diagonal_noise(cov::Union{Number,UniformScaling,Diagonal}) = cov
function _diagonal_noise(cov::AbstractMatrix)
    C = Matrix(cov)
    isdiag(C) || correlated_noise_error()
    return Diagonal(diag(C))
end

_selected_dims(::UniformScaling, o) = collect(1:o)
_selected_dims(::Diagonal, o) = collect(1:o)
function _selected_dims(M::AbstractMatrix, o)
    indices = Vector{Int}(undef, o)
    for k in 1:o
        nz = findall(!iszero, view(M, k, :))
        length(nz) == 1 || mixed_row_error(k)
        indices[k] = nz[1]
    end
    allunique(indices) || repeated_dims_error()
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
        m_tmp=m_tmp,
        x_tmp=x_tmp,
    )
end
# The block-wise `update!` only uses the first `o` blocks of the caches
make_obssized_cache(::BlockDiagonalCovariance, cache; o) = cache

nonpositive_noise_error() =
    throw(ArgumentError("The observation noise covariance must be positive definite."))
column_count_error(M, d) = throw(
    ArgumentError(
        "The observation matrix must have one column per ODE dimension (d = $d), but has " *
        "size $(size(M)). For second-order ODEs, it acts on `u` only."),
)
row_count_error(rows, o) = throw(
    DimensionMismatch(
        "The observation matrix maps the state to $rows values, but the data has $o " *
        "entries."),
)
too_many_rows_error(o, d) = throw(
    ArgumentError(
        "The observation matrix has $o rows, but at most d = $d (the ODE dimension) are " *
        "supported right now."),
)
zero_row_error(k) = throw(ArgumentError("Row $k of the observation matrix is zero."))
noise_size_error(noise_cov, o) = throw(
    DimensionMismatch(
        "The observation noise covariance must be $o×$o, one row and column per data " *
        "entry, but has size $(size(noise_cov))."),
)
kronecker_matrix_error() = throw(
    ArgumentError(
        "The isometric-kronecker covariance structure (e.g. of the default `EK0`) only " *
        "supports the observation matrix `I` or multiples of it. For other observation " *
        "matrices, e.g. partial observations, use a solver with a `DenseCovariance` or " *
        "`BlockDiagonalCovariance` covariance structure, like the `EK1` or `DiagonalEK1`.",
    ),
)
anisotropic_noise_error() = throw(
    ArgumentError(
        "Observations with the isometric-kronecker covariance structure (e.g. of the " *
        "default `EK0`) require isotropic observation noise, i.e. a multiple of the " *
        "identity."),
)
correlated_noise_error() = throw(
    ArgumentError(
        "Observations with a block-diagonal covariance structure require uncorrelated " *
        "observation noise, i.e. a diagonal noise covariance."),
)
mixed_row_error(k) = throw(
    ArgumentError(
        "Observation matrices with a block-diagonal covariance structure (e.g. the " *
        "`DiagonalEK1`) must select dimensions: each row must have exactly one nonzero " *
        "entry (any scaling is fine). Row $k mixes dimensions."),
)
repeated_dims_error() = throw(
    ArgumentError(
        "Observation matrices with a block-diagonal covariance structure (e.g. the " *
        "`DiagonalEK1`) must observe each dimension at most once. Got repeated " *
        "dimensions."),
)
