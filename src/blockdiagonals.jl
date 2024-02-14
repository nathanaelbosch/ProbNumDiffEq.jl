"""
BlockDiagonals.jl didn't cut it, so we're rolling our own.

TODO: Add a way to convert to a `BlockDiagonal`.
"""
struct MinimalAndFastBlockDiagonal{T<:Number, V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    function MinimalAndFastBlockDiagonal{T, V}(blocks::Vector{V}) where {T, V<:AbstractMatrix{T}}
        return new{T, V}(blocks)
    end
end
function MinimalAndFastBlockDiagonal(blocks::Vector{V}) where {T, V<:AbstractMatrix{T}}
    return MinimalAndFastBlockDiagonal{T, V}(blocks)
end
const MFBD = MinimalAndFastBlockDiagonal
blocks(B::MFBD) = B.blocks
nblocks(B::MFBD) = length(B.blocks)
size(B::MFBD) = (sum(size.(blocks(B), 1)), sum(size.(blocks(B), 2)))

function _block_indices(B::MFBD, i::Integer, j::Integer)
    all((0, 0) .< (i, j) .<= size(B)) || throw(BoundsError(B, (i, j)))
    # find the on-diagonal block `p` in column `j`
    p = 0
    @inbounds while j > 0
        p += 1
        j -= size(blocks(B)[p], 2)
    end
    # isempty to avoid reducing over an empty collection
    @views @inbounds i -= isempty(1:(p-1)) ? 0 : sum(size.(blocks(B)[1:(p-1)], 1))
    # if row `i` outside of block `p`, set `p` to place-holder value `-1`
    if i <= 0 || i > size(blocks(B)[p], 2)
        p = -1
    end
    return p, i, j
end
Base.@propagate_inbounds function Base.getindex(B::MFBD{T}, i::Integer, j::Integer) where T
    p, i, j = _block_indices(B, i, j)
    # if not in on-diagonal block `p` then value at `i, j` must be zero
    @inbounds return p > 0 ? blocks(B)[p][i, end + j] : zero(T)
end

Base.view(::MFBD, idxs...) =
    throw(ErrorException("`MinimalAndFastBlockDiagonal` does not support views!"))

copy(B::MFBD) = MFBD(copy.(blocks(B)))
copy!(B::MFBD, A::MFBD) = begin
    @assert length(A.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(B))
        copy!(B.blocks[i], A.blocks[i])
    end
    return B
end

_matmul!(
    C::MFBD{T},
    A::MFBD{T},
    B::MFBD{T},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], A.blocks[i], B.blocks[i])
    end
    return C
end

_matmul!(
    C::MFBD{T},
    A::MFBD{T},
    B::MFBD{T},
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
    C::MFBD{T},
    A::MFBD{T},
    B::Adjoint{T,<:MFBD{T}},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
    end
    return C
end

_matmul!(
    C::MFBD{T},
    A::Adjoint{T,<:MFBD{T}},
    B::MFBD{T},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.parent.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
    end
    return C
end

_matmul!(
    C::MFBD{T},
    A::MFBD{T},
    B::Adjoint{T, <:MFBD{T}},
) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
    @simd ivdep for i in eachindex(blocks(C))
        @inbounds _matmul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
    end
    return C
end

_matmul!(
    C::AbstractVector{T},
    A::MFBD{T},
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

LinearAlgebra.rmul!(B::MFBD, n::Number) = @simd ivdep for i in eachindex(B.blocks)
    rmul!(B.blocks[i], n)
end
LinearAlgebra.adjoint(B::MFBD) = Adjoint(B)

Base.:*(A::MFBD, B::MFBD) = begin
    @assert length(A.blocks) == length(B.blocks)
    return MFBD([blocks(A)[i] * blocks(B)[i] for i in eachindex(B.blocks)])
end
Base.:*(A::Adjoint{T,<:MFBD}, B::MFBD) where {T} = begin
    @assert length(A.parent.blocks) == length(B.blocks)
    return MFBD([A.parent.blocks[i]' * B.blocks[i] for i in eachindex(B.blocks)])
end
Base.:*(A::MFBD, B::Adjoint{T,<:MFBD}) where {T} = begin
    @assert length(A.blocks) == length(B.parent.blocks)
    return MFBD([A.blocks[i] * B.parent.blocks[i]' for i in eachindex(B.parent.blocks)])
end
Base.:*(A::UniformScaling, B::MFBD) = MFBD([A * blocks(B)[i] for i in eachindex(B.blocks)])
