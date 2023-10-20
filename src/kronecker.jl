mutable struct IsoKroneckerProduct{T<:Number,TB<:AbstractMatrix} <: Kronecker.AbstractKroneckerProduct{T}
    ldim::Int64
    B::TB
    function IsoKroneckerProduct(ldim::Int64, B::AbstractMatrix{T}) where {T}
        return new{T,typeof(B)}(ldim, B)
    end
end
IsoKroneckerProduct(ldim::Integer, B::AbstractVector) = IsoKroneckerProduct(ldim, reshape(B, :, 1))
const IKP = IsoKroneckerProduct

Kronecker.getmatrices(K::IKP) = (I(K.ldim), K.B)

function Base.:*(A::IKP, B::IKP)
    @assert A.ldim == B.ldim
    return IsoKroneckerProduct(A.ldim, A.B * B.B)
end
Base.:*(K::IKP, a::Number) = IsoKroneckerProduct(K.ldim, K.B * a)
Base.:*(a::Number, K::IKP) = IsoKroneckerProduct(K.ldim, a * K.B)
LinearAlgebra.adjoint(A::IKP) = IsoKroneckerProduct(A.ldim, A.B')

function check_same_size(A::IKP, B::IKP)
    if A.ldim != B.ldim || size(A.B) != size(B.B)
        Ad, An, Am, Bd, Bn, Bm = A.ldim, size(A)..., B.ldim, size(B)...
        throw(
            DimensionMismatch("A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm)"),
        )
    end
end
Base.:+(A::IKP, B::IKP) = begin
    check_same_size(A, B)
    return IsoKroneckerProduct(A.ldim, A.B + B.B)
end
Base.:+(U::UniformScaling, K::IKP) = IsoKroneckerProduct(K.ldim, U + K.B)
Base.:+(K::IKP, U::UniformScaling) = IsoKroneckerProduct(K.ldim, U + K.B)
Base.:-(U::UniformScaling, K::IKP) = IsoKroneckerProduct(K.ldim, U - K.B)
LinearAlgebra.inv(K::IKP) = IsoKroneckerProduct(K.ldim, inv(K.B))
Base.:/(A::IKP, B::IKP) = begin
    @assert A.ldim == B.ldim
    return IsoKroneckerProduct(A.ldim, A.B / B.B)
end
Base.:\(A::IKP, B::IKP) = begin
    @assert A.ldim == B.ldim
    return IsoKroneckerProduct(A.ldim, A.B / B.B)
end

_matmul!(A::IKP, B::IKP, C::IKP) = begin
    @assert A.ldim == B.ldim == C.ldim
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP{T}, B::IKP{T}, C::IKP{T}) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert A.ldim == B.ldim == C.ldim
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP, B::IKP, C::IKP, alpha::Number, beta::Number) = begin
    @assert A.ldim == B.ldim == C.ldim
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP{T}, B::IKP{T}, C::IKP{T}, alpha::Number, beta::Number
         ) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert A.ldim == B.ldim == C.ldim
    _matmul!(A.B, B.B, C.B, alpha, beta)
    return A
end
copy!(A::IKP, B::IKP) = begin
    @assert A.ldim == B.ldim
    copy!(A.B, B.B)
    return A
end
copy(A::IKP) = IsoKroneckerProduct(A.ldim, copy(A.B))
similar(A::IKP) = IsoKroneckerProduct(A.ldim, similar(A.B))
Base.size(K::IKP) = (K.ldim * size(K.B, 1), K.ldim * size(K.B, 2))

# conversion
Base.convert(::Type{T}, K::IKP) where {T<:IKP} =
    K isa T ? K : T(K)
function IKP{T,TB}(K::IKP) where {T,TB}
    IKP(K.ldim, convert(TB, K.B))
end

"""
Allocation-free reshape
Found here: https://discourse.julialang.org/t/convert-array-into-matrix-in-place/55624/5
"""
reshape_no_alloc(a, dims::Tuple) =
    invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
# reshape_no_alloc(a::AbstractArray, dims::Tuple) = reshape(a, dims)
reshape_no_alloc(a, dims...) = reshape_no_alloc(a, Tuple(dims))
reshape_no_alloc(a::Missing, dims::Tuple) = missing

function mul_vectrick!(x::AbstractVecOrMat, A::IsoKroneckerProduct, v::AbstractVecOrMat)
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    # @info "mul_vectrick!" typeof(x) typeof(A) typeof(v)
    # @info "mul_vectrick!" typeof(X) typeof(N) typeof(V)
    _matmul!(X, N, V)
    return x
end
function mul_vectrick!(
    x::AbstractVecOrMat,
    A::IsoKroneckerProduct,
    v::AbstractVecOrMat,
    alpha::Number,
    beta::Number,
    )
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    _matmul!(X, N, V, alpha, beta)
    return x
end

_matmul!(C::AbstractVecOrMat, A::IsoKroneckerProduct, B::AbstractVecOrMat) = mul_vectrick!(C, A, B)
mul!(C::AbstractMatrix, A::IsoKroneckerProduct, B::AbstractMatrix) = mul_vectrick!(C, A, B)
mul!(C::AbstractMatrix, A::IsoKroneckerProduct, B::Adjoint{T, <:AbstractMatrix{T}}) where {T} = mul_vectrick!(C, A, B)
mul!(C::AbstractVector, A::IsoKroneckerProduct, B::AbstractVector) = mul_vectrick!(C, A, B)

_matmul!(C::AbstractVecOrMat{T}, A::IsoKroneckerProduct{T}, B::AbstractVecOrMat{T}) where {T<:LinearAlgebra.BlasFloat} = mul_vectrick!(C, A, B)
_matmul!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::IsoKroneckerProduct) = _matmul!(C', B', A')
_matmul!(C::AbstractVecOrMat{T}, A::AbstractVecOrMat{T}, B::IsoKroneckerProduct{T}) where {T<:LinearAlgebra.BlasFloat} = _matmul!(C', B', A')

_matmul!(C::AbstractVecOrMat, A::IsoKroneckerProduct, B::AbstractVecOrMat, alpha::Number, beta::Number) = mul_vectrick!(C, A, B, alpha, beta)
_matmul!(C::AbstractVecOrMat{T}, A::IsoKroneckerProduct{T}, B::AbstractVecOrMat{T}, alpha::Number, beta::Number
         ) where {T<:LinearAlgebra.BlasFloat} = mul_vectrick!(C, A, B, alpha, beta)
_matmul!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::IsoKroneckerProduct, alpha::Number, beta::Number) = mul_vectrick!(C', B', A', alpha, beta)
_matmul!(C::AbstractVecOrMat{T}, A::AbstractVecOrMat{T}, B::IsoKroneckerProduct{T}, alpha::Number, beta::Number
         ) where {T<:LinearAlgebra.BlasFloat} = mul_vectrick!(C', B', A', alpha, beta)
