"""
    make_observation_model(cache, M, noise_cov; o)

Return `(H, R)` such that `o`-dimensional data `y` is modelled as `y = H x + N(0, R)`, with
`x` the filter state, `H = M E₀` and `R = noise_cov`, both in the covariance structure of
`cache`. Throw an error if `M` or `noise_cov` do not fit the data, or cannot be represented
in that covariance structure.
"""
function make_observation_model(cache, M, noise_cov; o)
    d, fac = cache.d, cache.covariance_factorization
    M = _structured(M)
    rows, cols = M isa UniformScaling ? (d, d) : size(M)
    cols == d || column_count_error(M, d)
    rows == o || row_count_error(rows, o)
    o <= d || too_many_rows_error(o, d)
    _has_zero_row(M) && zero_row_error()
    noise_cov isa AbstractMatrix && size(noise_cov) != (o, o) &&
        noise_size_error(noise_cov, o)
    _noise_fits(fac, noise_cov) || noise_structure_error(fac)
    H = solution_observation_matrix(fac, M)
    R = to_factorized_matrix(_measurement_structure(fac, o), cov2psdmatrix(noise_cov; d=o))
    return H, R
end

function check_observation_noise_cov(cov)
    _is_positive(cov) || nonpositive_noise_error()
    return nothing
end
_is_positive(cov::Number) = cov > 0
_is_positive(cov::UniformScaling) = cov.λ > 0
_is_positive(cov::Diagonal) = all(>(0), cov.diag)
_is_positive(cov::AbstractMatrix) = isposdef(Matrix(cov))

"""
    ScaledSelection(m, dims, d)

The `o×d` matrix `diag(m) * Π`, where `Π` consists of the rows `dims` of the `d×d` identity:
row `k` has its only nonzero entry `m[k]` in column `dims[k]`, and no column is used twice.
For `o = d` it is a generalized permutation matrix. As an observation matrix it observes
`m[k] * u[dims[k]]`.
"""
struct ScaledSelection{T,V<:AbstractVector{T}} <: AbstractMatrix{T}
    m::V
    dims::Vector{Int}
    d::Int
end
size(M::ScaledSelection) = (length(M.dims), M.d)
Base.getindex(M::ScaledSelection, k::Int, i::Int) =
    M.dims[k] == i ? M.m[k] : zero(eltype(M))

# `M` in its most structured type
_structured(M::Number) = M * I
_structured(M::UniformScaling) = M
_structured(M::Diagonal) = M
function _structured(M::AbstractMatrix)
    nonzeros = [findall(!iszero, row) for row in eachrow(M)]
    all(nz -> length(nz) == 1, nonzeros) || return M
    dims = only.(nonzeros)
    allunique(dims) || return M
    return ScaledSelection([M[k, i] for (k, i) in enumerate(dims)], dims, size(M, 2))
end

_has_zero_row(M::UniformScaling) = iszero(M.λ)
_has_zero_row(M::Diagonal) = any(iszero, M.diag)
_has_zero_row(::ScaledSelection) = false
_has_zero_row(M::AbstractMatrix) = any(iszero, eachrow(M))

_ei_row(::Type{T}, q, deriv, λ) where {T} = [zeros(T, 1, deriv) T(λ) zeros(T, 1, q - deriv)]

"""
    solution_observation_matrix(C::CovarianceStructure, M, deriv=0)

The observation matrix `H` that maps the filter state `x` to `M u⁽ⁱ⁾`, the `i`-th derivative
of the ODE solution times `M`, with `i = deriv`. It has the matrix type of the covariance
structure `C`, which limits `M`: a Kronecker `C` needs `M = λI`, and a block-diagonal `C` an
`M` that observes each dimension separately (`λI`, `Diagonal` or `ScaledSelection`).
"""
solution_observation_matrix(C::CovarianceStructure, M) =
    solution_observation_matrix(C, M, 0)
function solution_observation_matrix(C::DenseCovariance{T}, M, deriv) where {T}
    M = M isa UniformScaling ? Matrix{T}(M, C.d, C.d) : Matrix{T}(M)
    o = size(M, 1)
    return [zeros(T, o, C.d * deriv) M zeros(T, o, C.d * (C.q - deriv))]
end
# `eᵢᵀ ⊗ λI = (λ eᵢᵀ) ⊗ I_d`; no other `M` gives the form `B ⊗ I_d`
solution_observation_matrix(
    C::IsometricKroneckerCovariance{T}, M::UniformScaling, deriv,
) where {T} = IsometricKroneckerProduct(C.d, _ei_row(T, C.q, deriv, M.λ))
solution_observation_matrix(::IsometricKroneckerCovariance, M, deriv) =
    kronecker_matrix_error()
# `eᵢᵀ ⊗ Diagonal(m)` has the blocks `m[j] eᵢᵀ`
solution_observation_matrix(
    C::BlockDiagonalCovariance{T}, M::UniformScaling, deriv,
) where {T} = BlocksOfDiagonals([_ei_row(T, C.q, deriv, M.λ) for _ in 1:C.d])
solution_observation_matrix(C::BlockDiagonalCovariance{T}, M::Diagonal, deriv) where {T} =
    BlocksOfDiagonals([_ei_row(T, C.q, deriv, m) for m in M.diag])
solution_observation_matrix(
    C::BlockDiagonalCovariance{T}, M::ScaledSelection, deriv,
) where {T} =
    BlocksOfScaledSelections([_ei_row(T, C.q, deriv, m) for m in M.m], M.dims, C.d)
solution_observation_matrix(::BlockDiagonalCovariance, M, deriv) = selection_error()

_noise_fits(::DenseCovariance, noise_cov) = true
_noise_fits(::IsometricKroneckerCovariance, noise_cov) =
    noise_cov isa Union{Number,UniformScaling,Diagonal{<:Number,<:FillArrays.Fill}}
_noise_fits(::BlockDiagonalCovariance, noise_cov) =
    noise_cov isa Union{Number,UniformScaling,Diagonal}

_measurement_structure(C::BlockDiagonalCovariance{T}, o) where {T} =
    BlockDiagonalCovariance{T}(o, C.q)
_measurement_structure(C, o) = C

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
zero_row_error() = throw(ArgumentError("The observation matrix has a zero row."))
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
selection_error() = throw(
    ArgumentError(
        "The block-diagonal covariance structure (e.g. of the `DiagonalEK1`) requires " *
        "observation matrices that select dimensions: each row must have exactly one " *
        "nonzero entry (any scaling is fine), and no dimension may be observed twice."),
)
noise_structure_error(::IsometricKroneckerCovariance) = throw(
    ArgumentError(
        "The isometric-kronecker covariance structure (e.g. of the default `EK0`) " *
        "requires isotropic observation noise: a scalar variance, a `UniformScaling` or " *
        "an `Eye`."),
)
noise_structure_error(::BlockDiagonalCovariance) = throw(
    ArgumentError(
        "The block-diagonal covariance structure (e.g. of the `DiagonalEK1`) requires " *
        "uncorrelated observation noise: a scalar variance, a `UniformScaling` or a " *
        "`Diagonal`."),
)
