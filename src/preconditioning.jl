function init_preconditioner(d, q, ::Type{elType}=typeof(1.0)) where {elType}
    D = d * (q + 1)
    P = Diagonal(ones(elType, D))
    PI = Diagonal(ones(elType, D))
    return P, PI
end

function make_preconditioners!(cache::AbstractODEFilterCache, dt)
    @unpack P, PI, d, q = cache
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
    return nothing
end
function make_preconditioners!(post::GaussianODEFilterPosterior, dt)
    @unpack P, PI, d, q = post
    make_preconditioner!(P, dt, d, q)
    # make_preconditioner_inv!(PI, dt, d, q)
    PI.diag .= 1 ./ P.diag
    return nothing
end

@fastmath @inbounds function make_preconditioner!(P, h, d, q)
    val = factorial(q) / h^(q + 1 / 2)
    for j in 0:q
        @simd for i in 0:d-1
            P[j+i*(q+1)+1, j+i*(q+1)+1] = val
        end
        val /= (q - j) / h
    end
    return P
end

@fastmath @inbounds function make_preconditioner_inv!(PI, h, d, q)
    val = h^(q + 1 / 2) / factorial(q)
    for j in 0:q
        @simd for i in 0:d-1
            PI[j+i*(q+1)+1, j+i*(q+1)+1] = val
        end
        val *= (q - j) / h
    end
    return PI
end
