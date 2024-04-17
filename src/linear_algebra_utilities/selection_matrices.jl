struct SelectionMatrix{T<:Number} <: AbstractMatrix{T}
    dims_to_select::Vector{Int64}
    input_size::Int64
end
Base.size(A::SelectionMatrix) = (length(A.dims_to_select), A.input_size)
SelectionMatrix(dims_to_select, input_size) =
    SelectionMatrix{Bool}(dims_to_select, input_size)

# Base.@propagate_inbounds function Base.getindex(
#     M::SelectionMatrix{T},
#     i::Integer,
#     j::Integer,
# ) where {T}
#     all((0, 0) .< (i, j) .<= size(M)) || throw(BoundsError(M, (i, j)))

#     if j != M.dims_to_select[i]
#         return zero(T)
#     else
#         return one(T)
#     end
# end

Matrix(A::SelectionMatrix{T}) where {T} = begin
    M = zeros(T, size(A)...)
    for (i, d) in enumerate(A.dims_to_select)
        M[i, d] = one(T)
    end
    return M
end

Base.:*(M::SelectionMatrix, v::AbstractVector) =
    v[M.dims_to_select]
Base.:*(M::SelectionMatrix, A::AbstractMatrix) =
    A[M.dims_to_select, :]
Base.:*(M::Adjoint{<:Any,<:SelectionMatrix}, v::AbstractVector) = begin
    out = zeros(M.parent.input_size)
    out[M.parent.dims_to_select] = v
    return out
end
Base.:*(M::Adjoint{<:Any,<:SelectionMatrix}, A::AbstractMatrix) = begin
    out = similar(A, M.parent.input_size, size(A, 2))
    out .= 0
    out[M.parent.dims_to_select, :] .= A
    return out
end
Base.:*(A::AbstractMatrix, M::Adjoint{<:Any,<:SelectionMatrix}) = (M' * A')'
Base.:*(v::AbstractVector, M::Adjoint{<:Any,<:SelectionMatrix}) = (M' * v')'
Base.:*(A::Adjoint{<:Any,<:AbstractMatrix}, M::Adjoint{<:Any,<:SelectionMatrix}) =
    (M' * A')'
Base.:*(v::Adjoint{<:Any,<:AbstractVector}, M::Adjoint{<:Any,<:SelectionMatrix}) =
    (M' * v')'

Base.:*(M::SelectionMatrix, A::BlocksOfDiagonals) = begin
    @assert size(M, 2) == size(A, 1)
    @assert M.input_size == length(A.blocks)
end
