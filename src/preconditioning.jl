function init_preconditioner(d, q, ::Type{elType}=typeof(1.0)) where {elType}
    P = IsoKroneckerProduct(true, d, Diagonal(ones(elType, q + 1)))
    PI = IsoKroneckerProduct(true, d, Diagonal(ones(elType, q + 1)))
    return P, PI
end

function make_preconditioners!(cache::AbstractODEFilterCache, dt)
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

@fastmath @inbounds function make_preconditioner!(P::IsoKroneckerProduct, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    for j in 0:q
        P.B.diag[j+1] = val
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

@fastmath @inbounds function make_preconditioner_inv!(PI::IsoKroneckerProduct, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    for j in 0:q
        PI.B.diag[j+1] = val
        val *= (q - j) / h
    end
    return PI
end
