########################################################################################
# Integrated Brownian Motion
########################################################################################
"""Generate the discrete dynamics for a q-IBM model

Careful: Dimensions are ordered differently than in `probnum`!"""
function ibm(d::Integer, q::Integer; precond_dt=1.0)
    F̃ = diagm(1 => ones(q))
    I_d = diagm(0 => ones(d))
    F = kron(F̃, I_d)  # In probnum the order is inverted

    # L̃ = zeros(q+1)
    # L̃[end] = σ^2
    # I_d = diagm(0 => ones(d))
    # L = kron(L̃, I_d)'  # In probnum the order is inverted

    P, P_inv = preconditioner(precond_dt, d, q)


    @fastmath function A!(A::AbstractMatrix, h::Real)
        # Assumes that A comes from a previous computation => zeros and one-diag
        val = one(h)
        for i in 1:q
            val = val * h / (i)
            for j in 1:d*(q+1-i)
                @inbounds A[j,j+(d*i)] = val
            end
        end
        A .= P * A * P_inv
    end

    @fastmath function _transdiff_ibm_element(row::Int, col::Int, h::AbstractFloat)
        idx = 2 * q + 1 - row - col
        fact_rw = factorial(q - row)
        fact_cl = factorial(q - col)
        return (h^idx) / (idx * fact_rw * fact_cl)
    end
    @fastmath function Q!(Q::AbstractMatrix, h::AbstractFloat, σ²::AbstractFloat=1.0)
        val = one(h)
        @simd for col in 0:q
            @simd for row in col:q
                val = _transdiff_ibm_element(row, col, h)
                @simd for i in 0:d-1
                    @inbounds Q[1 + col*d + i,1 + row*d + i] = val * σ²
                    @inbounds Q[1 + row*d + i,1 + col*d + i] = val * σ²
                end
            end
        end
        Q .= P * Q * P'
    end

    return A!, Q!
end
