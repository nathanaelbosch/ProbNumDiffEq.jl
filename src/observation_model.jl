"""
    make_observation_model(cache, M, noise_cov; o)

Return `(H, R)` such that `o`-dimensional data `y` is modelled as `y = H x + N(0, R)`, with
`x` the filter state, `H = SolutionObservation(M)` and `R = noise_cov`, both in the
covariance structure of `cache`. Throw an error if `M` or `noise_cov` do not fit the data,
or can not be represented in that covariance structure.
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
    H = to_factorized_matrix(fac, SolutionObservation(M, cache.q))
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
    SolutionObservation(M, q, derivative=0)

The observation matrix `H = M * Eᵢ = eᵢᵀ ⊗ M` of the filter state, which observes the linear
combination `M * u⁽ⁱ⁾` of the `i`-th derivative of the ODE solution `u`, `i = derivative`.
Data observes `u`; the initializations observe higher derivatives.

With the derivative-major state `x = vec(X)`, `X ∈ ℝ^(d×(q+1))`, it is `H * x = M * X * eᵢ`.
So all structure of `H` is in `M`, which is a `UniformScaling`, a `Diagonal`, a
`ScaledSelection` or a general matrix. `M` determines which covariance structures can
represent `H`, see the `to_factorized_matrix` methods.
"""
struct SolutionObservation{MT}
    M::MT
    q::Int
    derivative::Int
end
SolutionObservation(M, q) = SolutionObservation(M, q, 0)

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

# `λ eᵢᵀ`, a row of length `q + 1`
_ei_row(::Type{T}, q, i, λ) where {T} = [zeros(T, 1, i) T(λ) zeros(T, 1, q - i)]

# Entries `i d + 1` to `(i + 1) d` of the derivative-major state are `u⁽ⁱ⁾`, so
# `H = [0 ⋯ 0 M 0 ⋯ 0]`
function to_factorized_matrix(C::DenseCovariance{T}, H::SolutionObservation) where {T}
    M = H.M isa UniformScaling ? Matrix{T}(H.M, C.d, C.d) : Matrix{T}(H.M)
    o, i = size(M, 1), H.derivative
    return [zeros(T, o, C.d * i) M zeros(T, o, C.d * (C.q - i))]
end
# `eᵢᵀ ⊗ λI = (λ eᵢᵀ) ⊗ I_d`; no other `M` gives the form `B ⊗ I_d`
to_factorized_matrix(
    C::IsometricKroneckerCovariance{T}, H::SolutionObservation{<:UniformScaling},
) where {T} = IsometricKroneckerProduct(C.d, _ei_row(T, C.q, H.derivative, H.M.λ))
to_factorized_matrix(::IsometricKroneckerCovariance, ::SolutionObservation) =
    kronecker_matrix_error()
# `eᵢᵀ ⊗ Diagonal(m)` has the blocks `m[j] eᵢᵀ`. A `ScaledSelection` is kept as it is: its
# row `k` reads only state block `dims[k]`, see the methods below and its update in
# `filtering/update.jl`.
to_factorized_matrix(
    C::BlockDiagonalCovariance{T}, H::SolutionObservation{<:UniformScaling},
) where {T} = BlocksOfDiagonals([_ei_row(T, C.q, H.derivative, H.M.λ) for _ in 1:C.d])
to_factorized_matrix(
    C::BlockDiagonalCovariance{T}, H::SolutionObservation{<:Diagonal},
) where {T} = BlocksOfDiagonals([_ei_row(T, C.q, H.derivative, m) for m in H.M.diag])
to_factorized_matrix(
    ::BlockDiagonalCovariance{T}, H::SolutionObservation{<:ScaledSelection},
) where {T} = SolutionObservation(
    ScaledSelection(T.(H.M.m), H.M.dims, H.M.d), H.q, H.derivative)
to_factorized_matrix(::BlockDiagonalCovariance, ::SolutionObservation) = selection_error()

_noise_fits(::DenseCovariance, noise_cov) = true
_noise_fits(::IsometricKroneckerCovariance, noise_cov) =
    noise_cov isa Union{Number,UniformScaling,Diagonal{<:Number,<:FillArrays.Fill}}
_noise_fits(::BlockDiagonalCovariance, noise_cov) =
    noise_cov isa Union{Number,UniformScaling,Diagonal}

_measurement_structure(C::BlockDiagonalCovariance{T}, o) where {T} =
    BlockDiagonalCovariance{T}(o, C.q)
_measurement_structure(C, o) = C

# Row `k` of a `ScaledSelection` observation is `m[k] eᵢᵀ` on state block `dims[k]`
_diagonal_block(H::SolutionObservation{<:ScaledSelection}, k, o) =
    _ei_row(eltype(H.M), H.q, H.derivative, H.M.m[k])
_matmul!(z::AbstractVector, H::SolutionObservation{<:ScaledSelection}, μ::AbstractVector) =
    z .= H.M.m .* view(μ, H.derivative * H.M.d .+ H.M.dims)

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
