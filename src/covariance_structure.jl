abstract type CovarianceStructure{T} end
struct IsometricKroneckerCovariance{T} <: CovarianceStructure{T}
    d::Int64
    q::Int64
end
struct DenseCovariance{T} <: CovarianceStructure{T}
    d::Int64
    q::Int64
end
struct BlockDiagonalCovariance{T} <: CovarianceStructure{T}
    d::Int64
    q::Int64
end

factorized_zeros(C::IsometricKroneckerCovariance{T}, sizes...) where {T} = begin
    for s in sizes
        @assert s % C.d == 0
    end
    return IsometricKroneckerProduct(C.d, Array{T}(calloc, (s ÷ C.d for s in sizes)...))
end
factorized_similar(C::IsometricKroneckerCovariance{T}, size1, size2) where {T} = begin
    for s in (size1, size2)
        @assert s % C.d == 0
    end
    return IsometricKroneckerProduct(C.d, similar(Matrix{T}, size1 ÷ C.d, size2 ÷ C.d))
end

factorized_zeros(::DenseCovariance{T}, sizes...) where {T} =
    Array{T}(calloc, sizes...)
factorized_similar(::DenseCovariance{T}, size1, size2) where {T} =
    similar(Matrix{T}, size1, size2)

factorized_zeros(C::BlockDiagonalCovariance{T}, sizes...) where {T} = begin
    for s in sizes
        @assert s % C.d == 0
    end
    return BlockDiagonal([Array{T}(calloc, (s ÷ C.d for s in sizes)...) for _ in 1:C.d])
end
factorized_similar(C::BlockDiagonalCovariance{T}, size1, size2) where {T} = begin
    for s in (size1, size2)
        @assert s % C.d == 0
    end
    return BlockDiagonal([similar(Matrix{T}, size1 ÷ C.d, size2 ÷ C.d) for _ in 1:C.d])
end

to_factorized_matrix(::DenseCovariance, M::AbstractMatrix) = Matrix(M)
to_factorized_matrix(::IsometricKroneckerCovariance, M::IsometricKroneckerProduct) = M
to_factorized_matrix(C::BlockDiagonalCovariance, M::IsometricKroneckerProduct) =
    BlockDiagonal([M.B for _ in 1:C.d])

for FT in [:DenseCovariance, :IsometricKroneckerCovariance, :BlockDiagonalCovariance]
    @eval to_factorized_matrix(FAC::$FT, M::PSDMatrix) =
        PSDMatrix(to_factorized_matrix(FAC, M.R))
end
