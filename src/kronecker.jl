mutable struct IsoKroneckerProduct{T<:Number,TA<:Number,TB<:AbstractMatrix} <: Kronecker.AbstractKroneckerProduct{T}
    alpha::TA
    ldim::Int64
    B::TB
    function IsoKroneckerProduct(alpha::T1, ldim::Int64, B::AbstractMatrix{T2}) where {T1,T2}
        return new{promote_type(T1,T2),typeof(alpha),typeof(B)}(alpha, ldim, B)
    end
end
IsoKroneckerProduct(alpha::Number, ldim::Integer, B::AbstractVector) = IsoKroneckerProduct(alpha, ldim, reshape(B, :, 1))
const IKP = IsoKroneckerProduct

Kronecker.getmatrices(K::IKP) = (K.alpha*I(K.ldim), K.B)

function Base.:*(A::IKP, B::IKP)
    @assert A.ldim == B.ldim
    return IsoKroneckerProduct(A.alpha * B.alpha, A.ldim, A.B * A.B)
end
Base.:*(K::IKP, a::Number) = IsoKroneckerProduct(K.alpha, K.ldim, K.B * a)
Base.:*(a::Number, K::IKP) = IsoKroneckerProduct(K.alpha, K.ldim, a * K.B)
LinearAlgebra.adjoint(A::IKP) = IsoKroneckerProduct(A.alpha, A.ldim, A.B')

_matmul!(A::IKP, B::IKP, C::IKP) = begin
    @assert A.ldim == B.ldim == C.ldim
    A.alpha = B.alpha * C.alpha
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP{T}, B::IKP{T}, C::IKP{T}) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert A.ldim == B.ldim == C.ldim
    A.alpha = B.alpha * C.alpha
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP, B::IKP, C::IKP, alpha::Number, beta::Number) = begin
    @assert A.ldim == B.ldim == C.ldim
    A.alpha = B.alpha * C.alpha
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP{T}, B::IKP{T}, C::IKP{T}, alpha::Number, beta::Number
         ) where {T<:LinearAlgebra.BlasFloat} = begin
    @assert A.ldim == B.ldim == C.ldim
    A.alpha = B.alpha * C.alpha
    _matmul!(A.B, B.B, C.B, alpha, beta)
    return A
end
copy!(A::IKP, B::IKP) = begin
    @assert A.ldim == B.ldim
    A.alpha = B.alpha
    copy!(A.B, B.B)
    return A
end
copy(A::IKP) = IsoKroneckerProduct(A.alpha, A.ldim, copy(A.B))
similar(A::IKP) = IsoKroneckerProduct(A.alpha, A.ldim, similar(A.B))
Base.size(K::IKP) = (K.ldim * size(K.B, 1), K.ldim * size(K.B, 2))

# conversion
Base.convert(::Type{T}, K::IKP) where {T<:IKP} =
    K isa T ? K : T(K)
function IKP{T,TA,TB}(K::IKP) where {T,TA,TB}
    IKP(convert(TA, K.alpha), K.ldim, convert(TB, K.B))
end

"""
Allocation-free reshape
Found here: https://discourse.julialang.org/t/convert-array-into-matrix-in-place/55624/5
"""
reshape_no_alloc(a, dims::Tuple) =
    invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
# reshape_no_alloc(a::AbstractArray, dims::Tuple) = reshape(a, dims)
reshape_no_alloc(a, dims...) = reshape_no_alloc(a, Tuple(dims))

function mul_vectrick!(x::AbstractVecOrMat, A::IsoKroneckerProduct, v::AbstractVecOrMat)
    N = A.B
    @assert A.alpha == 1
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
    @assert A.alpha == 1
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    _matmul!(X, N, V, alpha, beta)
    return x
end

_matmul!(C::AbstractVecOrMat, A::IsoKroneckerProduct, B::AbstractVecOrMat) = mul_vectrick!(C, A, B)
_matmul!(C::AbstractVecOrMat{T}, A::IsoKroneckerProduct{T}, B::AbstractVecOrMat{T}) where {T<:LinearAlgebra.BlasFloat} = mul_vectrick!(C, A, B)
_matmul!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::IsoKroneckerProduct) = _matmul!(C', B', A')
_matmul!(C::AbstractVecOrMat{T}, A::AbstractVecOrMat{T}, B::IsoKroneckerProduct{T}) where {T<:LinearAlgebra.BlasFloat} = _matmul!(C', B', A')

_matmul!(C::AbstractVecOrMat, A::IsoKroneckerProduct, B::AbstractVecOrMat, alpha::Number, beta::Number) = mul_vectrick!(C, A, B, alpha, beta)
_matmul!(C::AbstractVecOrMat{T}, A::IsoKroneckerProduct{T}, B::AbstractVecOrMat{T}, alpha::Number, beta::Number
         ) where {T<:LinearAlgebra.BlasFloat} = mul_vectrick!(C, A, B, alpha, beta)
_matmul!(C::AbstractVecOrMat, A::AbstractVecOrMat, B::IsoKroneckerProduct, alpha::Number, beta::Number) = mul_vectrick!(C', B', A', alpha, beta)
_matmul!(C::AbstractVecOrMat{T}, A::AbstractVecOrMat{T}, B::IsoKroneckerProduct{T}, alpha::Number, beta::Number
         ) where {T<:LinearAlgebra.BlasFloat} = mul_vectrick!(C', B', A', alpha, beta)
