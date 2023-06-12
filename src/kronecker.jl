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
copy(A::KP) = kronecker((A.A), copy(A.B))
