function preconditioner(elType, d, q)
    P_preallocated = Diagonal(ones(elType, d*(q+1)))

    @fastmath @inbounds function P(h)
        val = h^(-q-1/2)
        @simd for j in 0:q
            @simd for i in 1:d
                # P_preallocated[j*d + i,j*d + i] = h^(j-q-1/2)
                P_preallocated[j*d + i,j*d + i] = val
            end
            val *= h
        end
        return P_preallocated
    end

    return P
end
