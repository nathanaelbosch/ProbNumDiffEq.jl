"""
    BlocksOfDiagonals(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}

TODO
"""
struct BlocksOfDiagonals{T<:Number,V<:AbstractMatrix{T}} <: AbstractMatrix{T}
    blocks::Vector{V}
    function BlocksOfDiagonals{T,V}(
        blocks::Vector{V},
    ) where {T,V<:AbstractMatrix{T}}
        return new{T,V}(blocks)
    end
end
function BlocksOfDiagonals(blocks::Vector{V}) where {T,V<:AbstractMatrix{T}}
    return BlocksOfDiagonal{T,V}(blocks)
end

blocks(B::BlocksOfDiagonals) = B.blocks
nblocks(B::BlocksOfDiagonals) = length(B.blocks)
size(B::BlocksOfDiagonals) = mapreduce(size, ((a, b), (c, d)) -> (a + c, b + d), blocks(B))

Base.@propagate_inbounds function Base.getindex(
    B::BlocksOfDiagonals{T},
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

Base.view(::BlocksOfDiagonals, idxs...) =
    throw(ErrorException("`BlocksOfDiagonals` does not support views!"))

copy(B::BlocksOfDiagonals) = BlocksOfDiagonals(copy.(blocks(B)))
copy!(B::BlocksOfDiagonals, A::BlocksOfDiagonals) = begin
    @assert length(A.blocks) == length(B.blocks)
    @simd ivdep for i in eachindex(blocks(B))
        copy!(B.blocks[i], A.blocks[i])
    end
    return B
end
similar(B::BlocksOfDiagonals) = BlocksOfDiagonals(similar.(blocks(B)))
zero(B::BlocksOfDiagonals) = BlocksOfDiagonals(zero.(blocks(B)))

# Sums of BlocksOfDiagonals
Base.:+(A::BlocksOfDiagonals, B::BlocksOfDiagonals) = begin
    @assert nblocks(A) == nblocks(B)
    return BlocksOfDiagonals([Ai + Bi for (Ai, Bi) in zip(blocks(A), blocks(B))])
end
Base.:-(A::BlocksOfDiagonals, B::BlocksOfDiagonals) = begin
    @assert nblocks(A) == nblocks(B)
    return BlocksOfDiagonals([Ai - Bi for (Ai, Bi) in zip(blocks(A), blocks(B))])
end

add!(out::BlocksOfDiagonals, toadd::BlocksOfDiagonals) = begin
    @assert nblocks(out) == nblocks(toadd)
    @simd ivdep for i in eachindex(blocks(out))
        add!(blocks(out)[i], blocks(toadd)[i])
    end
    return out
end

# Mul with Scalar or UniformScaling
Base.:*(a::Number, M::BlocksOfDiagonals) = BlocksOfDiagonals([a * B for B in blocks(M)])
Base.:*(M::BlocksOfDiagonals, a::Number) = BlocksOfDiagonals([B * a for B in blocks(M)])
Base.:*(U::UniformScaling, M::BlocksOfDiagonals) = BlocksOfDiagonals([U * B for B in blocks(M)])
Base.:*(M::BlocksOfDiagonals, U::UniformScaling) = BlocksOfDiagonals([B * U for B in blocks(M)])

# Mul between BockDiag's
Base.:*(A::BlocksOfDiagonals, B::BlocksOfDiagonals) = begin
    @assert length(A.blocks) == length(B.blocks)
    return BlocksOfDiagonals([Ai * Bi for (Ai, Bi) in zip(blocks(A), blocks(B))])
end
Base.:*(A::Adjoint{T,<:BlocksOfDiagonals}, B::BlocksOfDiagonals) where {T} = begin
    @assert length(A.parent.blocks) == length(B.blocks)
    return BlocksOfDiagonals([Ai' * Bi for (Ai, Bi) in zip(blocks(A.parent), blocks(B))])
end
Base.:*(A::BlocksOfDiagonals, B::Adjoint{T,<:BlocksOfDiagonals}) where {T} = begin
    @assert length(A.blocks) == length(B.parent.blocks)
    return BlocksOfDiagonals([Ai * Bi' for (Ai, Bi) in zip(blocks(A), blocks(B.parent))])
end

# Standard LinearAlgebra.mul!
for _mul! in (:mul!, :_matmul!)
    @eval $_mul!(C::BlocksOfDiagonals, A::BlocksOfDiagonals, B::BlocksOfDiagonals) = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i])
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlocksOfDiagonals{T},
        A::BlocksOfDiagonals{T},
        B::BlocksOfDiagonals{T},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i])
        end
        return C
    end

    @eval $_mul!(C::BlocksOfDiagonals, A::BlocksOfDiagonals, B::BlocksOfDiagonals, alpha::Number, beta::Number) =
        begin
            @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
            @simd ivdep for i in eachindex(blocks(C))
                @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i], alpha, beta)
            end
            return C
        end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlocksOfDiagonals{T},
        A::BlocksOfDiagonals{T},
        B::BlocksOfDiagonals{T},
        alpha::Number,
        beta::Number,
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B.blocks[i], alpha, beta)
        end
        return C
    end

    @eval $_mul!(C::BlocksOfDiagonals, A::Adjoint{<:Number,<:BlocksOfDiagonals}, B::BlocksOfDiagonals) = begin
        @assert length(C.blocks) == length(A.parent.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlocksOfDiagonals{T},
        A::BlocksOfDiagonals{T},
        B::Adjoint{T,<:BlocksOfDiagonals{T}},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
        end
        return C
    end

    @eval $_mul!(C::BlocksOfDiagonals, A::BlocksOfDiagonals, B::Adjoint{<:Number,<:BlocksOfDiagonals}) = begin
        @assert length(C.blocks) == length(A.blocks) == length(B.parent.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], adjoint(B.parent.blocks[i]))
        end
        return C
    end
    (_mul! == :_matmul!) && @eval $_mul!(
        C::BlocksOfDiagonals{T},
        A::Adjoint{T,<:BlocksOfDiagonals{T}},
        B::BlocksOfDiagonals{T},
    ) where {T<:LinearAlgebra.BlasFloat} = begin
        @assert length(C.blocks) == length(A.parent.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], adjoint(A.parent.blocks[i]), B.blocks[i])
        end
        return C
    end

    @eval $_mul!(C::BlocksOfDiagonals, A::Number, B::BlocksOfDiagonals) = begin
        @assert length(C.blocks) == length(B.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A, B.blocks[i])
        end
        return C
    end
    @eval $_mul!(C::BlocksOfDiagonals, A::BlocksOfDiagonals, B::Number) = begin
        @assert length(C.blocks) == length(A.blocks)
        @simd ivdep for i in eachindex(blocks(C))
            @inbounds $_mul!(C.blocks[i], A.blocks[i], B)
        end
        return C
    end

    @eval $_mul!(
        C::AbstractVector,
        A::BlocksOfDiagonals,
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
        A::BlocksOfDiagonals{T},
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

LinearAlgebra.rmul!(B::BlocksOfDiagonals, n::Number) = begin
    @simd ivdep for i in eachindex(B.blocks)
        rmul!(B.blocks[i], n)
    end
    return B
end

LinearAlgebra.inv(A::BlocksOfDiagonals) = BlocksOfDiagonals(inv.(blocks(A)))

copy!(A::BlocksOfDiagonals, B::Diagonal) = begin
    @assert size(A) == size(B)
    i = 1
    for Ai in blocks(A)
        d = LinearAlgebra.checksquare(Ai)
        @views copy!(Ai, Diagonal(B.diag[i:i+d-1]))
        i += d
    end
    return A
end

Base.:*(D::Diagonal, A::BlocksOfDiagonals) = begin
    @assert size(D, 2) == size(A, 1)
    local i = 1
    outblocks = map(blocks(A)) do Ai
        d = size(Ai, 1)
        outi = Diagonal(view(D.diag, i:(i+d-1))) * Ai
        i += d
        outi
    end
    return BlocksOfDiagonals(outblocks)
end
Base.:*(A::BlocksOfDiagonals, D::Diagonal) = begin
    local i = 1
    outblocks = map(blocks(A)) do Ai
        d = size(Ai, 2)
        outi = Ai * Diagonal(view(D.diag, i:(i+d-1)))
        i += d
        outi
    end
    return BlocksOfDiagonals(outblocks)
end
for _mul! in (:mul!, :_matmul!)
    @eval $_mul!(C::BlocksOfDiagonals, A::BlocksOfDiagonals, B::Diagonal) = begin
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
    @eval $_mul!(C::BlocksOfDiagonals, A::Diagonal, B::BlocksOfDiagonals) = begin
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
    @eval $_mul!(C::BlocksOfDiagonals, A::BlocksOfDiagonals, B::Diagonal, alpha::Number, beta::Number) =
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
    @eval $_mul!(C::BlocksOfDiagonals, A::Diagonal, B::BlocksOfDiagonals, alpha::Number, beta::Number) =
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

Base.isequal(A::BlocksOfDiagonals, B::BlocksOfDiagonals) =
    length(A.blocks) == length(B.blocks) && all(map(isequal, A.blocks, B.blocks))
==(A::BlocksOfDiagonals, B::BlocksOfDiagonals) =
    length(A.blocks) == length(B.blocks) && all(map(==, A.blocks, B.blocks))
