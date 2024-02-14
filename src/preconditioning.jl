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
    P = MFBD([Diagonal(ones(elType, C.q + 1)) for _ in 1:C.d])
    PI = MFBD([Diagonal(ones(elType, C.q + 1)) for _ in 1:C.d])
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

@fastmath @inbounds function make_preconditioner!(P::IsometricKroneckerProduct, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    @simd ivdep for j in 0:q
        P.B.diag[j+1] = val
        val /= (q - j) / h
    end
    return P
end

@fastmath @inbounds function make_preconditioner!(P::MFBD, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    @simd ivdep for j in 0:q
        for M in P.blocks
            M.diag[j+1] = val
        end
        val /= (q - j) / h
    end
    return P
end

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

@fastmath @inbounds function make_preconditioner_inv!(
    PI::IsometricKroneckerProduct, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    @simd ivdep for j in 0:q
        PI.B.diag[j+1] = val
        val *= (q - j) / h
    end
    return PI
end

@fastmath @inbounds function make_preconditioner_inv!(
    PI::MFBD, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    @simd ivdep for j in 0:q
        for M in PI.blocks
            M.diag[j+1] = val
        end
        val *= (q - j) / h
    end
    return PI
end
