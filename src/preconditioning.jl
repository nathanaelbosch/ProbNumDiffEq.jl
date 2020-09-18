function preconditioner(d, q)
    P_preallocated = Diagonal(zeros(d*(q+1), d*(q+1)))
    Pinv_preallocated = Diagonal(zeros(d*(q+1), d*(q+1)))

    @fastmath @inbounds function P(h)
        @simd for i in 1:d
            @simd for j in 0:q
                P_preallocated[j*d + i,j*d + i] = h^(j-q)
            end
        end
        return P_preallocated
    end

    @fastmath @inbounds function P_inv(h)
        @simd for i in 1:d
            @simd for j in 0:q
                Pinv_preallocated[j*d + i,j*d + i] = h^(q-j)
            end
        end
        return Pinv_preallocated
    end

    return (P=P, P_inv=P_inv)
end
