function init_preconditioner(d, q, ::Type{elType}=typeof(1.0)) where {elType}
    Id = _I(d)
    P = kronecker(Id, Diagonal(ones(elType, q + 1)))
    PI = kronecker(Id, Diagonal(ones(elType, q + 1)))
    return P, PI
end

function make_preconditioners!(cache::AbstractODEFilterCache, dt)
    @unpack P, PI, d, q = cache
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return nothing
end

@fastmath @inbounds function make_preconditioner!(P, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    for j in 0:q
        P.B.diag[j+1] = val
        val /= (q - j) / h
    end
    return P
end

@fastmath @inbounds function make_preconditioner_inv!(PI, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    for j in 0:q
        PI.B.diag[j+1] = val
        val *= (q - j) / h
    end
    return PI
end
