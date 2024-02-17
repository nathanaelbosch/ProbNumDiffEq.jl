function init_preconditioner(C::IsometricKroneckerCovariance{elType}) where {elType}
    P = IsometricKroneckerProduct(C.d, Diagonal(ones(elType, C.q + 1)))
    PI = IsometricKroneckerProduct(C.d, Diagonal(ones(elType, C.q + 1)))
    return P, PI
end
function init_preconditioner(C::DenseCovariance{elType}) where {elType}
    P = kron(I(C.d), Diagonal(ones(elType, C.q + 1)))
    PI = kron(I(C.d), Diagonal(ones(elType, C.q + 1)))
    return P, PI
end
function init_preconditioner(C::BlockDiagonalCovariance{elType}) where {elType}
    B = Diagonal(ones(elType, C.q + 1))
    P = BlockDiag([B for _ in 1:C.d])
    BI = Diagonal(ones(elType, C.q + 1))
    PI = BlockDiag([BI for _ in 1:C.d])
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

@fastmath @inbounds function make_preconditioner!(P::Diagonal, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    for j in 0:q
        @simd ivdep for i in 0:d-1
            # P[j+i*(q+1)+1, j+i*(q+1)+1] = val
            P.diag[j+i*(q+1)+1] = val
        end
        val /= (q - j) / h
    end
    return P
end
make_preconditioner!(P::IsometricKroneckerProduct, h, d, q) =
    (make_preconditioner!(P.B, h, 1, q); P)
make_preconditioner!(P::BlockDiag, h, d, q) =
    (make_preconditioner!(blocks(P)[1], h, 1, q); P)

@fastmath @inbounds function make_preconditioner_inv!(PI::Diagonal, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    for j in 0:q
        @simd ivdep for i in 0:d-1
            # PI[j+i*(q+1)+1, j+i*(q+1)+1] = val
            PI.diag[j+i*(q+1)+1] = val
        end
        val *= (q - j) / h
    end
    return PI
end
make_preconditioner_inv!(PI::IsometricKroneckerProduct, h, d, q) =
    (make_preconditioner_inv!(PI.B, h, 1, q); PI)
make_preconditioner_inv!(PI::BlockDiag, h, d, q) =
    (make_preconditioner_inv!(blocks(PI)[1], h, 1, q); PI)
