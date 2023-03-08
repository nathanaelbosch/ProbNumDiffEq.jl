"""
    _matmul!(C, A, B)

Efficiently compute `C = A * B` in-place.

This function is not exported and is only used internally. Essentially, depending on the
type of the input matrices, it either calls `LinearAlgebra.mul!`, `Octavian.matmul!`, or
if some matrices are `Diagonal` it does some broadcasting stuff with FastBroadcase.jl.
"""
_matmul!(C, A, B)

# By default use mul!
_matmul!(C, A, B) = mul!(C, A, B)
_matmul!(C, A, B, a, b) = mul!(C, A, B, a, b)
# Some special cases
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractMatrix{T},
    B::Diagonal{T},
) where {T<:LinearAlgebra.BlasFloat} = (@.. C = A * B.diag')
_matmul!(
    C::AbstractMatrix{T},
    A::Diagonal{T},
    B::AbstractMatrix{T},
) where {T<:LinearAlgebra.BlasFloat} = (@.. C = A.diag * B)
_matmul!(
    C::AbstractMatrix{T},
    A::Diagonal{T},
    B::Diagonal{T},
) where {T<:LinearAlgebra.BlasFloat} = @.. C = A * B
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
    alpha::Number,
    beta::Number,
) where {T<:LinearAlgebra.BlasFloat} = matmul!(C, A, B, alpha, beta)
_matmul!(
    C::AbstractMatrix{T},
    A::AbstractVecOrMat{T},
    B::AbstractVecOrMat{T},
) where {T<:LinearAlgebra.BlasFloat} = matmul!(C, A, B)

"""
    getupperright(A)

Get the upper right part of a matrix `A` without allocating.

This function is mostly there to make accessing `R` from a `QR` object more efficient, as
`qr!(A).R` allocates since it does not use views. This function is not exported and is only
ever used internally by `triangularize!`(@ref).
"""
function getupperright(A)
    m, n = size(A)
    return triu!(@view A[1:min(m, n), 1:n])
end

"""
    triangularize!(A; [cachemat])

Compute `qr(A).R` in the most efficient and allocation-free way possible.

The fallback implementation essentially computes `qr!(A).R`. But if `A` is of type
`StridedMatrix{<:LinearAlgebra.BlasFloat}`, we can make things more efficient by calling
LAPACK directly and using the preallocated cache `cachemat`.
"""
triangularize!(A; cachemat)

function triangularize!(A; cachemat=nothing)
    QR = qr!(A)
    return getupperright(getfield(QR, :factors))
end
function triangularize!(A::StridedMatrix{<:LinearAlgebra.BlasFloat}; cachemat)
    A, _ = LinearAlgebra.LAPACK.geqrt!(A, cachemat)
    return getupperright(A)
end

"""
    fast_X_A_Xt!(out::PSDMatrix, A::PSDMatrix, X::AbstractMatrix)

Compute `out .= X * A * X'` in-place, more efficiently than with `PSDMatrices.X_A_Xt!`.

This function just reimplements the simple `PSDMatrices.X_A_Xt!` with `_matmul!`(@ref)
instead of `LinearAlgebra.mul!`.
"""
function fast_X_A_Xt!(out::PSDMatrix, A::PSDMatrix, X::AbstractMatrix)
    _matmul!(out.R, A.R, X')
    return out
end
