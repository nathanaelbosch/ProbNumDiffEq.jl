"""
A type for positive semi-definite matrices.

Relates to PSDMatrices.jl, and I might move this to PSDMatrices.jl in the
future, but having this here allowed for easier development.
"""

struct SquarerootMatrix{T<:Real,S<:AbstractMatrix,M<:AbstractMatrix} <: AbstractMatrix{T}
    squareroot::S
    mat::M
    SquarerootMatrix(S::AbstractMatrix{T}, mat::AbstractMatrix{T}) where {T} =
        new{T,typeof(S),typeof(mat)}(S, mat)
end
SquarerootMatrix(S) = SquarerootMatrix(S, S * S')

Base.Matrix(M::SquarerootMatrix) = M.mat
# TODO Maybe cache the above, into M.mat or something. But do so lazily!

Base.size(M::SquarerootMatrix) = size(M.mat)
# getindex is expensive, since each call performs a matrix multiplication
Base.getindex(M::SquarerootMatrix, I::Vararg{Int,N}) where {N} = getindex(M.mat, I...)
Base.copy(M::SquarerootMatrix) = SquarerootMatrix(
    # I want the copy to look exactly as the original one
    M.squareroot isa LinearAlgebra.Adjoint ? copy(M.squareroot')' : copy(M.squareroot),
    copy(M.mat),
)
Base.copy!(dst::SquarerootMatrix, src::SquarerootMatrix) =
    (Base.copy!(dst.squareroot, src.squareroot);
    Base.copy!(dst.mat, src.mat);
    dst)
Base.similar(M::SquarerootMatrix) = SquarerootMatrix(similar(M.squareroot), similar(M.mat))

X_A_Xt(M::SquarerootMatrix, X::AbstractMatrix) = SquarerootMatrix(X * M.squareroot)
X_A_Xt!(out::SquarerootMatrix, M::SquarerootMatrix, X::AbstractMatrix) = begin
    _matmul!(out.squareroot, X, M.squareroot)
    _matmul!(out.mat, out.squareroot, out.squareroot')
    return out
end
X_A_Xt!(out::SquarerootMatrix, M::SquarerootMatrix, D::Diagonal) = begin
    # Basically just to optimize PI*Q*PI
    out.squareroot .= D.diag .* M.squareroot
    out.mat .= D.diag .* M.mat .* D.diag'
    return out
end

Base.inv(M::SquarerootMatrix) = Base.inv(M.mat)
LinearAlgebra.diag(M::SquarerootMatrix) = LinearAlgebra.diag(M.mat)
LinearAlgebra.cholesky(M::SquarerootMatrix) = LinearAlgebra.cholesky(M.mat)
LinearAlgebra.cholesky!(M::SquarerootMatrix) = LinearAlgebra.cholesky!(M.mat)
LinearAlgebra.logdet(M::SquarerootMatrix) = LinearAlgebra.logdet(M.mat)
