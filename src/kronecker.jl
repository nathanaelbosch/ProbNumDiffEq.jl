# Piracy
const KP = Kronecker.KroneckerProduct
_matmul!(A::KP, B::KP, C::KP) = begin
    _matmul!(A.A, B.A, C.A)
    _matmul!(A.B, B.B, C.B)
    return A
end
_matmul!(A::KP, B::KP, C::KP, a, b) = begin
    _matmul!(A.A, B.A, C.A, a, b)
    _matmul!(A.B, B.B, C.B, a, b)
    return A
end
copy!(A::KP, B::KP) = begin
    copy!(A.A, B.A)
    copy!(A.B, B.B)
    return A
end
copy(A::KP) = kronecker(copy(A.A), copy(A.B))

"""
    _I(d) = I(d) * I(d)

Create an identity matrix that does not change its type when multiplied by another identity matrix.

# Examples
```julia-repl
julia> I(2)|> typeof
Diagonal{Bool, Vector{Bool}}

julia> I(2) * I(2) |> typeof
Diagonal{Bool, BitVector}

julia> _I(2) |> typeof
Diagonal{Bool, BitVector}

julia> _I(2) * _I(2) |> typeof
Diagonal{Bool, BitVector}
```
"""
_I(d) = I(d) * I(d)

# Isometric Kronecker products
const IsoKronecker{T,M1,M2} = Kronecker.KroneckerProduct{T,M1,M2} where {T,M1<:Diagonal,M2}

"""
Allocation-free reshape
Found here: https://discourse.julialang.org/t/convert-array-into-matrix-in-place/55624/5
"""
reshape_no_alloc(a, dims::Tuple) = invoke(Base._reshape, Tuple{AbstractArray,typeof(dims)}, a, dims)
reshape_no_alloc(a, dims...) = reshape_no_alloc(a, Tuple(dims))

function Kronecker.mul_vec_trick!(x::AbstractVector, A::IsoKronecker, v::AbstractVector)
    M, N = getmatrices(A)
    a, b = size(M)
    c, d = size(N)

    V = reshape_no_alloc(v, (d, b))
    X = reshape_no_alloc(x, (c, a))
    if b * c * (a + d) < a * d * (b + c)
        _matmul!(V, V, transpose(M))
        _matmul!(X, N, V)
    else
        _matmul!(X, N, V)
        _matmul!(X, X, transpose(M))
    end
    return x
end
