Base.view(::BlockDiagonal, idxs...) =
    throw(MethodError("BlockDiagonal does not support views"))

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
    C::BlockDiagonal{T},
    A::BlockDiagonal{T},
    B::Adjoint{T, <:BlockDiagonal{T}},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
    end
    return C
end

_matmul!(
    C::BlockDiagonal{T},
    A::Adjoint{T, <:BlockDiagonal{T}},
    B::BlockDiagonal{T},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.parent.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
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

function BlockDiagonals.isequal_blocksizes(B1::BlockDiagonal, B2::BlockDiagonal)
    @assert length(B1.blocks) == length(B2.blocks)
    for i in eachindex(B1.blocks)
        if size(B1.blocks[i]) != size(B2.blocks[i])
            return false
        end
    end
    return true
end

LinearAlgebra.adjoint(B::BlockDiagonal) = Adjoint(B)
