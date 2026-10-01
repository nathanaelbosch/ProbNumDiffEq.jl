"""
    on_kronecker_factors(f, d, args...; kwargs...)

Call the dense operation `f` once, on the Kronecker factors of all arguments, and return
its result.

With [`IsometricKroneckerCovariance`](@ref), every matrix has the form `B ⊗ I_d`, so the
filter is a single dense filter on the factors `B`, whose means have one column per ODE
dimension. Matrices become `B`, vectors of length `n d` become the `n × d` matrix whose rows
are the derivatives, `PSDMatrix`, `Gaussian` and `AffineNormalKernel` are converted field by
field, and scalars, `nothing` and `missing` pass through.
"""
on_kronecker_factors(f, d, args...; kwargs...) =
    _on_kronecker_factors(f, d, args, values(kwargs))

# Converting inside the varargs method (here and in `foreach_diagonal_block`) does not
# infer for mixed argument types, so `f` would be dispatched at runtime and allocate
_on_kronecker_factors(f, d, args, kwargs) = f(
    map(x -> _kronecker_factor(x, d), args)...;
    map(x -> _kronecker_factor(x, d), kwargs)...,
)

_kronecker_factor(M::IsometricKroneckerProduct, d) = M.B
_kronecker_factor(v::AbstractVector, d) = reshape(v, d, :)'
_kronecker_factor(M::PSDMatrix, d) = PSDMatrix(_kronecker_factor(M.R, d))
_kronecker_factor(x::Gaussian, d) =
    Gaussian(_kronecker_factor(x.μ, d), _kronecker_factor(x.Σ, d))
_kronecker_factor(K::AffineNormalKernel, d) = AffineNormalKernel(
    _kronecker_factor(K.A, d), _kronecker_factor(K.b, d), _kronecker_factor(K.C, d))
_kronecker_factor(x::Union{Number,Nothing,Missing}, d) = x

"""
    foreach_diagonal_block(f, d, args...; kwargs...)

Call the dense operation `f` once for each of the `d` blocks, on that block of all
arguments.

With [`BlockDiagonalCovariance`](@ref), every matrix is a [`BlocksOfDiagonals`](@ref) and
the filter decouples into `d` independent dense filters, one per ODE dimension. In the
`i`-th call, matrices become their `i`-th block, vectors the strided view `x[i:d:end]`, a
`Diagonal` (a per-dimension diffusion) its `i`-th entry, `PSDMatrix`, `Gaussian` and
`AffineNormalKernel` are converted field by field, and scalars, `nothing` and `missing`
pass through.
"""
function foreach_diagonal_block(f, d, args...; kwargs...)
    for i in 1:d
        _on_diagonal_block(f, i, d, args, values(kwargs))
    end
    return nothing
end

_on_diagonal_block(f, i, d, args, kwargs) = f(
    map(x -> _diagonal_block(x, i, d), args)...;
    map(x -> _diagonal_block(x, i, d), kwargs)...,
)

_diagonal_block(M::BlocksOfDiagonals, i, d) = blocks(M)[i]
_diagonal_block(v::AbstractVector, i, d) = view(v, i:d:length(v))
_diagonal_block(D::Diagonal, i, d) = D.diag[i]
_diagonal_block(M::PSDMatrix, i, d) = PSDMatrix(_diagonal_block(M.R, i, d))
_diagonal_block(x::Gaussian, i, d) =
    Gaussian(_diagonal_block(x.μ, i, d), _diagonal_block(x.Σ, i, d))
_diagonal_block(K::AffineNormalKernel, i, d) = AffineNormalKernel(
    _diagonal_block(K.A, i, d), _diagonal_block(K.b, i, d), _diagonal_block(K.C, i, d))
_diagonal_block(x::Union{Number,Nothing,Missing}, i, d) = x
