@fastmath @inbounds function make_preconditioner!(P, h, d, q)
    val = h^(-q-1/2)
    @simd for j in 0:q
        @simd for i in 1:d
            P[j*d + i,j*d + i] = val
        end
        val *= h
    end
    return P
end


@fastmath @inbounds function make_preconditioner_inv!(PI, h, d, q)
    val = h^(q+1/2)
    @simd for j in 0:q
        @simd for i in 1:d
            PI[j*d + i,j*d + i] = val
        end
        val /= h
    end
    return PI
end


function make_preconditioners!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK}, dt)
    @unpack P, PI, d, q = integ.cache
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
end
function make_preconditioners!(post::GaussianODEFilterPosterior, dt)
    @unpack P, PI, d, q = post
    make_preconditioner!(P, dt, d, q)
    make_preconditioner_inv!(PI, dt, d, q)
end
