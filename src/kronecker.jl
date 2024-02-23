copy(K::Kronecker.KroneckerProduct) = Kronecker.KroneckerProduct(copy(K.A), copy(K.B))
@doc raw"""
    IsometricKroneckerProduct(left_factor_dim::Int64, left_factor::AbstractMatrix)

Kronecker product of an identity and a generic matrix:
```math
\begin{aligned}
K = B \otimes I_d
\end{aligned}
```

# Arguments
- `right_factor_dim::Int64`: Dimension `d` of the left identity kronecker factor.
- `left_factor::AbstractMatrix`: Right Kronecker factor.
"""
struct RightIsometricKroneckerProduct{T<:Number,TB<:AbstractMatrix} <:
       Kronecker.AbstractKroneckerProduct{T}
    rdim::Int64
    B::TB
    function RightIsometricKroneckerProduct(
        right_factor_dim::Int64,
        left_factor::AbstractMatrix{T},
    ) where {T}
        return new{T,typeof(left_factor)}(right_factor_dim, left_factor)
    end
end
RightIsometricKroneckerProduct(rdim::Integer, B::AbstractVector) =
    RightIsometricKroneckerProduct(rdim, reshape(B, :, 1))
RightIsometricKroneckerProduct(M::AbstractMatrix) = throw(
    ArgumentError(
        "Can not create RightIsometricKroneckerProduct from the provided matrix of type $(typeof(M))",
    ),
)

const IsometricKroneckerProduct = RightIsometricKroneckerProduct
const IKP = IsometricKroneckerProduct

Kronecker.getmatrices(K::IKP) = (K.B, Eye(K.rdim))
Kronecker.getallfactors(K::IKP) = (K.B, Eye(K.rdim))

Base.zero(A::IKP) = IsometricKroneckerProduct(A.rdim, zero(A.B))
Base.one(A::IKP) = IsometricKroneckerProduct(A.rdim, one(A.B))
copy!(A::IKP, B::IKP) = begin
    check_same_size(A, B)
    copy!(A.B, B.B)
    return A
end
copy(A::IKP) = IsometricKroneckerProduct(A.rdim, copy(A.B))
similar(A::IKP) = IsometricKroneckerProduct(A.rdim, similar(A.B))
Base.size(K::IKP) = (K.rdim * size(K.B, 1), K.rdim * size(K.B, 2))

# conversion
Base.convert(::Type{T}, K::IKP) where {T<:IKP} =
    K isa T ? K : T(K)
function IKP{T,TB}(K::IKP) where {T,TB}
    IKP(K.rdim, convert(TB, K.B))
end

function Base.:*(A::IKP, B::IKP)
    @assert A.rdim == B.rdim
    return IsometricKroneckerProduct(A.rdim, A.B * B.B)
end
Base.:*(K::IKP, a::Number) = IsometricKroneckerProduct(K.rdim, K.B * a)
Base.:*(a::Number, K::IKP) = IsometricKroneckerProduct(K.rdim, a * K.B)
LinearAlgebra.adjoint(A::IKP) = IsometricKroneckerProduct(A.rdim, A.B')
LinearAlgebra.rmul!(A::IKP, b::Number) = IsometricKroneckerProduct(A.rdim, rmul!(A.B, b))

function check_same_size(A::IKP, B::IKP)
    if A.rdim != B.rdim || size(A.B) != size(B.B)
        Ad, An, Am, Bd, Bn, Bm = A.rdim, size(A)..., B.rdim, size(B)...
        throw(
            DimensionMismatch("A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm)"),
        )
    end
end
function check_matmul_sizes(A::IKP, B::IKP)
    # For A * B
    Ad, Bd = A.rdim, B.rdim
    An, Am, Bn, Bm = size(A)..., size(B)...
    if !(A.rdim == B.rdim) || !(Am == Bn)
        throw(
            DimensionMismatch(
                "Matrix multiplication not compatible: A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm)",
            ),
        )
    end
end
function check_matmul_sizes(C::IKP, A::IKP, B::IKP)
    # For C = A * B
    Ad, Bd, Cd = A.rdim, B.rdim, C.rdim
    An, Am, Bn, Bm, Cn, Cm = size(A)..., size(B)..., size(C)...
    if !(A.rdim == B.rdim == C.rdim) || !(Am == Bn && An == Cn && Bm == Cm)
        throw(
            DimensionMismatch(
                "Matrix multiplication not compatible: A has size ($Ad⋅$An,$Ad⋅$Am), B has size ($Bd⋅$Bn,$Bd⋅$Bm), C has size ($Cd⋅$Cn,$Cd⋅$Cm)",
            ),
        )
    end
end

Base.:+(A::IKP, B::IKP) = begin
    check_same_size(A, B)
    return IsometricKroneckerProduct(A.rdim, A.B + B.B)
end
Base.:+(U::UniformScaling, K::IKP) = IsometricKroneckerProduct(K.rdim, U + K.B)
Base.:+(K::IKP, U::UniformScaling) = IsometricKroneckerProduct(K.rdim, U + K.B)

add!(out::IsometricKroneckerProduct, toadd::IsometricKroneckerProduct) = begin
    @assert out.rdim == toadd.rdim
    add!(out.B, toadd.B)
    return out
end

Base.:-(U::UniformScaling, K::IKP) = IsometricKroneckerProduct(K.rdim, U - K.B)
LinearAlgebra.inv(K::IKP) = IsometricKroneckerProduct(K.rdim, inv(K.B))
Base.:/(A::IKP, B::IKP) = begin
    @assert A.rdim == B.rdim
    return IsometricKroneckerProduct(A.rdim, A.B / B.B)
end
Base.:\(A::IKP, B::IKP) = begin
    @assert A.rdim == B.rdim
    return IsometricKroneckerProduct(A.rdim, A.B \ B.B)
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

function _prepare_inputs_for_vectrick(A, x, v)
    M = A.B
    a, b = size(M)
    V = reshape_no_alloc(transpose(v), (length(v) ÷ b, b)) |> transpose
    X = reshape_no_alloc(transpose(x), (length(x) ÷ a, a)) |> transpose
    return X, V
end

function mul_vectrick!(x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat)
    X, V = _prepare_inputs_for_vectrick(A, x, v)
    mul!(X, A.B, V)
    return x
end
function mul_vectrick!(
    x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat, alpha::Number, beta::Number)
    X, V = _prepare_inputs_for_vectrick(A, x, v)
    mul!(X, A.B, V, alpha, beta)
    return x
end

function _matmul_vectrick!(x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat)
    X, V = _prepare_inputs_for_vectrick(A, x, v)
    _matmul!(X, A.B, V)
    return x
end
function _matmul_vectrick!(
    x::AbstractVecOrMat, A::IKP, v::AbstractVecOrMat, alpha::Number, beta::Number)
    X, V = _prepare_inputs_for_vectrick(A, x, v)
    _matmul!(X, A.B, V, alpha, beta)
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
    X, V = _prepare_inputs_for_vectrick(A, x, v)
    copyto!(X, A.B \ V)
    return x
end
