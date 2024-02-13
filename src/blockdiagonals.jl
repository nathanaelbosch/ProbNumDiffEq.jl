
_matmul!(
    C::BlockDiagonal{T},
    A::BlockDiagonal{T},
    B::BlockDiagonal{T},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], A.blocks[i], B.blocks[i])
    end
    return C
end

_matmul!(
    C::BlockDiagonal{T},
    A::BlockDiagonal{T},
    B::BlockDiagonal{T},
    alpha::Number,
    beta::Number,
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], A.blocks[i], B.blocks[i], alpha, beta)
    end
    return C
end

_matmul!(
    C::AbstractVector{T},
    A::BlockDiagonal{T},
    B::AbstractVector{T},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert size(A, 2) == length(B)
    @assert length(C) == size(A, 1)
    ic, ib = 1, 1
    for i in eachindex(blocks(A))
        d1, d2 = size(A.blocks[i])
        @inbounds _matmul!(view(C, ic:(ic+d1-1)), A.blocks[i], view(B, ib:(ib+d2-1)))
        ic += d1
        ib += d2
    end
    return C
end

function LinearAlgebra.cholesky!(B::BlockDiagonal)
    C = BlockDiagonal(map(b -> parent(UpperTriangular(cholesky!(b).U)), blocks(B)))
    return Cholesky(C, 'U', 0)
end
