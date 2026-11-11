function init_preconditioner(
    C::Union{IsometricKroneckerCovariance{elType},BlockDiagonalCovariance{elType}},
) where {elType}
    P = IsometricKroneckerProduct(C.d, Diagonal(ones(elType, C.q + 1)))
    PI = IsometricKroneckerProduct(C.d, Diagonal(ones(elType, C.q + 1)))
    return P, PI
end
function init_preconditioner(C::DenseCovariance{elType}) where {elType}
    P = kron(Diagonal(ones(elType, C.q + 1)), Eye(C.d))
    PI = kron(Diagonal(ones(elType, C.q + 1)), Eye(C.d))
    return P, PI
end

function make_preconditioners!(cache, dt)
    @unpack P, PI, d, q = cache
    return make_preconditioners!(P, PI, d, q, dt)
end

function make_preconditioners!(P, PI, d, q, dt)
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return nothing
end

@fastmath function make_preconditioner!(P::Diagonal, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    for j in 0:q
        @simd ivdep for i in 1:d
            P.diag[j*d+i] = val
        end
        val /= (q - j) / h
    end
    return P
end
make_preconditioner!(P::IsometricKroneckerProduct, h, d, q) =
    (make_preconditioner!(P.B, h, 1, q); P)

@fastmath @inbounds function make_preconditioner_inv!(PI::Diagonal, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    for j in 0:q
        @simd ivdep for i in 1:d
            PI.diag[j*d+i] = val
        end
        val *= (q - j) / h
    end
    return PI
end
make_preconditioner_inv!(PI::IsometricKroneckerProduct, h, d, q) =
    (make_preconditioner_inv!(PI.B, h, 1, q); PI)
