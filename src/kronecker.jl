copy(K::Kronecker.KroneckerProduct) = Kronecker.KroneckerProduct(copy(K.A), copy(K.B))
@doc raw"""
    IsometricKroneckerProduct(left_factor_dim::Int64, right_factor::AbstractMatrix)

Kronecker product of an identity and a generic matrix:
```math
\begin{aligned}
K = I_d \otimes B
\end{aligned}
```

# Arguments
- `left_factor_dim::Int64`: Dimension `d` of the left identity kronecker factor.
- `right_factor::AbstractMatrix`: Right Kronecker factor.
"""
struct IsometricKroneckerProduct{T<:Number,TB<:AbstractMatrix} <:
       Kronecker.AbstractKroneckerProduct{T}
    ldim::Int64
    B::TB
    function IsometricKroneckerProduct(
        left_factor_dim::Int64,
        right_factor::AbstractMatrix{T},
    ) where {T}
        return new{T,typeof(right_factor)}(left_factor_dim, right_factor)
    end
end
IsometricKroneckerProduct(ldim::Integer, B::AbstractVector) =
    IsometricKroneckerProduct(ldim, reshape(B, :, 1))
IsometricKroneckerProduct(M::AbstractMatrix) = throw(
    ArgumentError(
        "Can not create IsometricKroneckerProduct from the provided matrix of type $(typeof(M))",
    ),
)

const IKP = IsometricKroneckerProduct

Kronecker.getmatrices(K::IKP) = (I(K.ldim), K.B)
Kronecker.getallfactors(K::IKP) = (I(K.ldim), K.B)

Base.zero(A::IKP) = IsometricKroneckerProduct(A.ldim, zero(A.B))
Base.one(A::IKP) = IsometricKroneckerProduct(A.ldim, one(A.B))
copy!(A::IKP, B::IKP) = begin
    check_same_size(A, B)
    copy!(A.B, B.B)
    return A
end
copy(A::IKP) = IsometricKroneckerProduct(A.ldim, copy(A.B))
similar(A::IKP) = IsometricKroneckerProduct(A.ldim, similar(A.B))
Base.size(K::IKP) = (K.ldim * size(K.B, 1), K.ldim * size(K.B, 2))

# conversion
Base.convert(::Type{T}, K::IKP) where {T<:IKP} =
    K isa T ? K : T(K)
function IKP{T,TB}(K::IKP) where {T,TB}
    IKP(K.ldim, convert(TB, K.B))
end

function Base.:*(A::IKP, B::IKP)
    @assert A.ldim == B.ldim
    return IsometricKroneckerProduct(A.ldim, A.B * B.B)
end
Base.:*(K::IKP, a::Number) = IsometricKroneckerProduct(K.ldim, K.B * a)
Base.:*(a::Number, K::IKP) = IsometricKroneckerProduct(K.ldim, a * K.B)
LinearAlgebra.adjoint(A::IKP) = IsometricKroneckerProduct(A.ldim, A.B')
LinearAlgebra.rmul!(A::IKP, b::Number) = IsometricKroneckerProduct(A.ldim, rmul!(A.B, b))

function check_same_size(A::IKP, B::IKP)
    if A.ldim != B.ldim || size(A.B) != size(B.B)
        Ad, An, Am, Bd, Bn, Bm = A.ldim, size(A)..., B.ldim, size(B)...
        throw(
            DimensionMismatch("A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm)"),
        )
    end
end
function check_matmul_sizes(A::IKP, B::IKP)
    # For A * B
    Ad, Bd = A.ldim, B.ldim
    An, Am, Bn, Bm = size(A)..., size(B)...
    if !(A.ldim == B.ldim) || !(Am == Bn)
        throw(
            DimensionMismatch(
                "Matrix multiplication not compatible: A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm)",
            ),
        )
    end
end
function check_matmul_sizes(C::IKP, A::IKP, B::IKP)
    # For C = A * B
    Ad, Bd, Cd = A.ldim, B.ldim, C.ldim
    An, Am, Bn, Bm, Cn, Cm = size(A)..., size(B)..., size(C)...
    if !(A.ldim == B.ldim == C.ldim) || !(Am == Bn && An == Cn && Bm == Cm)
        throw(
            DimensionMismatch(
                "Matrix multiplication not compatible: A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm), C has size ($Cd⋅$Cn,$Cd⋅$Cm)",
            ),
        )
    end
end

Base.:+(A::IKP, B::IKP) = begin
    check_same_size(A, B)
    return IsometricKroneckerProduct(A.ldim, A.B + B.B)
end
Base.:+(U::UniformScaling, K::IKP) = IsometricKroneckerProduct(K.ldim, U + K.B)
Base.:+(K::IKP, U::UniformScaling) = IsometricKroneckerProduct(K.ldim, U + K.B)

add!(out::IsometricKroneckerProduct, toadd::IsometricKroneckerProduct) = begin
    @assert out.ldim == toadd.ldim
    add!(out.B, toadd.B)
end

Base.:-(U::UniformScaling, K::IKP) = IsometricKroneckerProduct(K.ldim, U - K.B)
LinearAlgebra.inv(K::IKP) = IsometricKroneckerProduct(K.ldim, inv(K.B))
Base.:/(A::IKP, B::IKP) = begin
    @assert A.ldim == B.ldim
    return IsometricKroneckerProduct(A.ldim, A.B / B.B)
end
Base.:\(A::IKP, B::IKP) = begin
    @assert A.ldim == B.ldim
    return IsometricKroneckerProduct(A.ldim, A.B \ B.B)
end

mul!(A::IKP, B::IKP, C::IKP) = begin
    check_matmul_sizes(A, B, C)
    mul!(A.B, B.B, C.B)
    return A
end
mul!(A::IKP, B::IKP, C::IKP, alpha::Number, beta::Number) = begin
    check_matmul_sizes(A, B, C)
    mul!(A.B, B.B, C.B, alpha, beta)
    return A
end

# fast_linalg.jl
_matmul!(A::IKP, B::IKP, C::IKP) = begin
    check_matmul_sizes(A, B, C)
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP{T}, B::IKP{T}, C::IKP{T}) where {T<:LinearAlgebra.BlasFloat} = begin
    check_matmul_sizes(A, B, C)
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::IKP, B::IKP, C::IKP, alpha::Number, beta::Number) = begin
    check_matmul_sizes(A, B, C)
    _matmul!(A.B, B.B, C.B, alpha, beta)
    return A
end
_matmul!(
    A::IKP{T},
    B::IKP{T},
    C::IKP{T},
    alpha::Number,
    beta::Number,
) where {T<:LinearAlgebra.BlasFloat} = begin
    check_matmul_sizes(A, B, C)
    _matmul!(A.B, B.B, C.B, alpha, beta)
    return A
end

mul!(A::IKP, b::Number, C::IKP) = begin
    check_matmul_sizes(A, C)
    mul!(A.B, b, C.B)
    return A
end
mul!(A::IKP, B::IKP, c::Number) = begin
    check_matmul_sizes(A, B)
    mul!(A.B, B.B, c)
    return A
end
_matmul!(A::IKP, b::Number, C::IKP) = begin
    check_matmul_sizes(A, C)
    _matmul!(A.B, b, C.B)
    return A
end
_matmul!(A::IKP, B::IKP, c::Number) = begin
    check_matmul_sizes(A, B)
    _matmul!(A.B, B.B, c)
    return A
end

"""
Allocation-free reshape
Found here: https://discourse.julialang.org/t/convert-array-into-matrix-in-place/55624/5
"""
reshape_no_alloc(a, dims::Tuple) =
    invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
reshape_no_alloc(a, dims...) = reshape_no_alloc(a, Tuple(dims))
reshape_no_alloc(a::Missing, dims::Tuple) = missing

function mul_vectrick!(x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat)
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    mul!(X, N, V)
    return x
end
function mul_vectrick!(
    x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat, alpha::Number, beta::Number)
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    mul!(X, N, V, alpha, beta)
    return x
end

function _matmul_vectrick!(x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat)
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    _matmul!(X, N, V)
    return x
end
function _matmul_vectrick!(
    x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat, alpha::Number, beta::Number)
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (d, length(v) ÷ d))
    X = reshape_no_alloc(x, (c, length(x) ÷ c))
    _matmul!(X, N, V, alpha, beta)
    return x
end

# mul! as mul_vectrick!
for TC in [:AbstractVector, :AbstractMatrix]
    @eval mul!(C::$TC, A::IKP, B::$TC) = mul_vectrick!(C, A, B)
    @eval mul!(C::$TC, A::IKP, B::Adjoint{T,<:$TC{T}}) where {T} = mul_vectrick!(C, A, B)
    @eval mul!(C::$TC, A::IKP, B::$TC, alpha::Number, beta::Number) =
        mul_vectrick!(C, A, B, alpha, beta)
end

# fast_linalg.jl
for TC in [:AbstractVector, :AbstractMatrix]
    @eval _matmul!(C::$TC, A::IKP, B::$TC) = _matmul_vectrick!(C, A, B)
    @eval _matmul!(
        C::$TC{T},
        A::IKP{T},
        B::$TC{T},
    ) where {T<:LinearAlgebra.BlasFloat} = _matmul_vectrick!(C, A, B)
    @eval _matmul!(C::$TC, A::$TC, B::IKP) = _matmul!(C', B', A')'
    @eval _matmul!(
        C::$TC{T},
        A::$TC{T},
        B::IKP{T},
    ) where {T<:LinearAlgebra.BlasFloat} = _matmul!(C', B', A')'

    @eval _matmul!(C::$TC, A::IKP, B::$TC, alpha::Number, beta::Number) =
        _matmul_vectrick!(C, A, B, alpha, beta)
    @eval _matmul!(
        C::$TC{T},
        A::IKP{T},
        B::$TC{T},
        alpha::Number,
        beta::Number,
    ) where {T<:LinearAlgebra.BlasFloat} = _matmul_vectrick!(C, A, B, alpha, beta)
    @eval _matmul!(C::$TC, A::$TC, B::IKP, alpha::Number, beta::Number) =
        _matmul_vectrick!(C', B', A', alpha, beta)'
    @eval _matmul!(
        C::$TC{T},
        A::$TC{T},
        B::IKP{T},
        alpha::Number,
        beta::Number,
    ) where {T<:LinearAlgebra.BlasFloat} = _matmul_vectrick!(C', B', A', alpha, beta)'
end

function Kronecker.ldiv_vec_trick!(x::AbstractVector, A::IKP, v::AbstractVector)
    N = A.B
    c, d = size(N)

    V = reshape_no_alloc(v, (c, length(v) ÷ c))
    X = reshape_no_alloc(x, (d, length(x) ÷ d))
    copyto!(X, N \ V)
    return x
end
