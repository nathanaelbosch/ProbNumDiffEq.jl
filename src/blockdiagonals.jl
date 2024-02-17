"""
    ProbNumDiffEqBlockDiagonal(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}

A very minimal but fast re-implementation of `BlockDiagonals.Blockdiagonal`.
"""
struct ProbNumDiffEqBlockDiagonal{T<:Number,V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    function ProbNumDiffEqBlockDiagonal{T,V}(
        blocks::Vector{V},
    ) where {T,V<:AbstractMatrix{T}}
        return new{T,V}(blocks)
    end
end
function ProbNumDiffEqBlockDiagonal(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}
    return ProbNumDiffEqBlockDiagonal{T,V}(blocks)
end
const BlockDiag = ProbNumDiffEqBlockDiagonal

blocks(B::BlockDiag) = B.blocks
nblocks(B::BlockDiag) = length(B.blocks)
size(B::BlockDiag) = mapreduce(size, ((a, b), (c, d)) -> (a + c, b + d), blocks(B))

Base.@propagate_inbounds function Base.getindex(
    B::BlockDiag{T},
    i::Integer,
    j::Integer,
) where {T}
    all((0, 0) .< (i, j) .<= size(B)) || throw(BoundsError(B, (i, j)))

    p = 1
    Si, Sj = size(blocks(B)[p])
    while p <= nblocks(B)
        if i <= Si && j <= Sj
            return blocks(B)[p][i, j]
        elseif (i <= Si && j > Sj) || (j <= Sj && i > Si)
            return zero(T)
        else
            i -= Si
            j -= Sj
            p += 1
        end
    end
    error("This shouldn't happen")
end

Base.view(::BlockDiag, idxs...) =
    throw(ErrorException("`BlockDiag` does not support views!"))

copy(B::BlockDiag) = BlockDiag(copy.(blocks(B)))
copy!(B::BlockDiag, A::BlockDiag) = begin
    @assert length(A.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(B))
        copy!(B.blocks[i], A.blocks[i])
    end
    return B
end
similar(B::BlockDiag) = BlockDiag(similar.(blocks(B)))
zero(B::BlockDiag) = BlockDiag(zero.(blocks(B)))

# Mul with Scalar or UniformScaling
Base.:*(a::Number, M::BlockDiag) = BlockDiag([a * B for B in blocks(M)])
Base.:*(M::BlockDiag, a::Number) = BlockDiag([B * a for B in blocks(M)])
Base.:*(U::UniformScaling, M::BlockDiag) = BlockDiag([U * B for B in blocks(M)])
Base.:*(M::BlockDiag, U::UniformScaling) = BlockDiag([B * U for B in blocks(M)])

# Mul between BockDiag's
Base.:*(A::BlockDiag, B::BlockDiag) = begin
    @assert length(A.blocks) == length(B.blocks)
    return BlockDiag([Ai * Bi for (Ai, Bi) in zip(blocks(A), blocks(B))])
end
Base.:*(A::Adjoint{T,<:BlockDiag}, B::BlockDiag) where {T} = begin
    @assert length(A.parent.blocks) == length(B.blocks)
    return BlockDiag([Ai' * Bi for (Ai, Bi) in zip(blocks(A.parent), blocks(B))])
end
Base.:*(A::BlockDiag, B::Adjoint{T,<:BlockDiag}) where {T} = begin
    @assert length(A.blocks) == length(B.parent.blocks)
    return BlockDiag([Ai * Bi' for (Ai, Bi) in zip(blocks(A), blocks(B.parent))])
end

# Standard LinearAlgebra.mul!
for _mul! in (:mul!, :_matmul!)
    @eval $_mul!(C::BlockDiag, A::BlockDiag, B::BlockDiag) = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i])
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlockDiag{T},
        A::BlockDiag{T},
        B::BlockDiag{T},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i])
        end
        return C
    end

    @eval $_mul!(C::BlockDiag, A::BlockDiag, B::BlockDiag, alpha::Number, beta::Number) =
        begin
            @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
            @simd ivdep for i in eachindex(blocks(C))
                @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i], alpha, beta)
            end
            return C
        end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlockDiag{T},
        A::BlockDiag{T},
        B::BlockDiag{T},
        alpha::Number,
        beta::Number,
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i], alpha, beta)
        end
        return C
    end

    @eval $_mul!(C::BlockDiag, A::Adjoint{<:Number,<:BlockDiag}, B::BlockDiag) = begin
        @assert length(C.blocks) == length(A.parent.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlockDiag{T},
        A::BlockDiag{T},
        B::Adjoint{T,<:BlockDiag{T}},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
        end
        return C
    end

    @eval $_mul!(C::BlockDiag, A::BlockDiag, B::Adjoint{<:Number,<:BlockDiag}) = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlockDiag{T},
        A::Adjoint{T,<:BlockDiag{T}},
        B::BlockDiag{T},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.parent.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
        end
        return C
    end

    @eval $_mul!(C::BlockDiag, A::Number, B::BlockDiag) = begin
        @assert length(C.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A, B.blocks[i])
        end
        return C
    end
    @eval $_mul!(C::BlockDiag, A::BlockDiag, B::Number) = begin
        @assert length(C.blocks) == length(A.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B)
        end
        return C
    end

    @eval $_mul!(
        C::AbstractVector,
        A::BlockDiag,
        B::AbstractVector,
    ) = begin
        @assert size(A, 2) == length(B)
        @assert length(C) == size(A, 1)
        ic, ib = 1, 1
        for i in eachindex(blocks(A))
            d1, d2 = size(A.blocks[i])
            @inbounds $_mul!(view(C, ic:(ic+d1-1)), A.blocks[i], view(B, ib:(ib+d2-1)))
            ic += d1
            ib += d2
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::AbstractVector{T},
        A::BlockDiag{T},
        B::AbstractVector{T},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert size(A, 2) == length(B)
        @assert length(C) == size(A, 1)
        ic, ib = 1, 1
        for i in eachindex(blocks(A))
            d1, d2 = size(A.blocks[i])
            @inbounds $_mul!(view(C, ic:(ic+d1-1)), A.blocks[i], view(B, ib:(ib+d2-1)))
            ic += d1
            ib += d2
        end
        return C
    end
end

LinearAlgebra.rmul!(B::BlockDiag, n::Number) = begin
    @simd ivdep for i in eachindex(B.blocks)
        rmul!(B.blocks[i], n)
    end
    return B
end

LinearAlgebra.inv(A::BlockDiag) = BlockDiag(inv.(blocks(A)))

copy!(A::BlockDiag, B::Diagonal) = begin
    @assert size(A) == size(B)
    i = 1
    for Ai in blocks(A)
        d = LinearAlgebra.checksquare(Ai)
        @views copy!(Ai, Diagonal(B.diag[i:i+d-1]))
        i += d
    end
    return A
end

Base.:*(D::Diagonal, A::BlockDiag) = begin
    @assert size(D, 2) == size(A, 1)
    local i = 1
    outblocks = map(blocks(A)) do Ai
        d = size(Ai, 1)
        outi = Diagonal(view(D.diag, i:(i+d-1))) * Ai
        i += d
        outi
    end
    return BlockDiag(outblocks)
end
Base.:*(A::BlockDiag, D::Diagonal) = begin
    local i = 1
    outblocks = map(blocks(A)) do Ai
        d = size(Ai, 2)
        outi = Ai * Diagonal(view(D.diag, i:(i+d-1)))
        i += d
        outi
    end
    return BlockDiag(outblocks)
end
for _mul! in (:mul!, :_matmul!)
    @eval $_mul!(C::BlockDiag, A::BlockDiag, B::Diagonal) = begin
        local i = 1
        @assert nblocks(C) == nblocks(A)
        for j in eachindex(blocks(C))
            Ci, Ai = blocks(C)[j], blocks(A)[j]
            d = size(Ai, 2)
            $_mul!(Ci, Ai, Diagonal(view(B.diag, i:(i+d-1))))
            i += d
        end
        return C
    end
    @eval $_mul!(C::BlockDiag, A::Diagonal, B::BlockDiag) = begin
        local i = 1
        @assert nblocks(C) == nblocks(B)
        for j in eachindex(blocks(C))
            Ci, Bi = blocks(C)[j], blocks(B)[j]
            d = size(Bi, 1)
            $_mul!(Ci, Diagonal(view(A.diag, i:(i+d-1))), Bi)
            i += d
        end
        return C
    end
    @eval $_mul!(C::BlockDiag, A::BlockDiag, B::Diagonal, alpha::Number, beta::Number) =
        begin
            local i = 1
            @assert nblocks(C) == nblocks(A)
            for j in eachindex(blocks(C))
                Ci, Ai = blocks(C)[j], blocks(A)[j]
                d = size(Ai, 2)
                $_mul!(Ci, Ai, Diagonal(view(B.diag, i:(i+d-1))), alpha, beta)
                i += d
            end
            return C
        end
    @eval $_mul!(C::BlockDiag, A::Diagonal, B::BlockDiag, alpha::Number, beta::Number) =
        begin
            i = 1
            @assert nblocks(C) == nblocks(B)
            for j in eachindex(blocks(C))
                Ci, Bi = blocks(C)[j], blocks(B)[j]
                d = size(Bi, 1)
                @inbounds $_mul!(Ci, Diagonal(view(A.diag, i:(i+d-1))), Bi, alpha, beta)
                i += d
            end
            return C
        end
end

Base.isequal(A::BlockDiag, B::BlockDiag) =
    length(A.blocks) == length(B.blocks) && all(map(isequal, A.blocks, B.blocks))
==(A::BlockDiag, B::BlockDiag) =
    length(A.blocks) == length(B.blocks) && all(map(==, A.blocks, B.blocks))
