########################################################################################
# Integrated Brownian Motion
########################################################################################
"""Generate the discrete dynamics for a q-IBM model. INCLUDES AUTOMATIC PRECONDITIONING!

Careful: Dimensions are ordered differently than in `probnum`!"""
function ibm(d::Integer, q::Integer, elType=typeof(1.0))
    F̃ = diagm(1 => ones(q))
    I_d = diagm(0 => ones(d))
    F = kron(F̃, I_d)  # In probnum the order is inverted

    A_base = diagm(0=>ones(elType, d*(q+1)))
    Q_base = zeros(elType, d*(q+1), d*(q+1))

    @fastmath function A!(A::AbstractMatrix, h::Real)
        # Assumes that A comes from a previous computation => zeros and one-diag
        val = one(h)
        for i in 1:q
            val = val / i
            for j in 1:d*(q+1-i)
                @inbounds A[j,j+(d*i)] = val
            end
        end
    end
    A!(A_base, 1.0)
    A_base = UpperTriangular(A_base)

    @fastmath function _transdiff_ibm_element(row::Int, col::Int, h::Real)
        idx = 2 * q + 1 - row - col
        fact_rw = factorial(q - row)
        fact_cl = factorial(q - col)
        return h / (idx * fact_rw * fact_cl)
    end
    @fastmath function Q!(Q::AbstractMatrix, h::Real, σ²::Real=1.0)
        val = one(h)
        @simd for col in 0:q
            @simd for row in col:q
                val = _transdiff_ibm_element(row, col, h) * σ²
                @simd for i in 0:d-1
                    @inbounds Q[1 + col*d + i,1 + row*d + i] = val
                    @inbounds Q[1 + row*d + i,1 + col*d + i] = val
                end
            end
        end
    end

    Q!(Q_base, 1.0)
    QR = UpperTriangular(collect(cholesky(Q_base).L'))
    Q_psd = PSDMatrix(QR)

    return A_base, Q_psd
end


"""Same as above, but without the automatic preconditioning"""
function vanilla_ibm(d::Integer, q::Integer)
    F̃ = diagm(1 => ones(q))
    I_d = diagm(0 => ones(d))
    F = kron(F̃, I_d)  # In probnum the order is inverted

    @fastmath function A!(A::AbstractMatrix, h::Real)
        # Assumes that A comes from a previous computation => zeros and one-diag
        val = one(h)
        for i in 1:q
            val = val * h / i
            for j in 1:d*(q+1-i)
                @inbounds A[j,j+(d*i)] = val
            end
        end
    end

    @fastmath function _transdiff_ibm_element(row::Int, col::Int, h::Real)
        idx = 2 * q + 1 - row - col
        fact_rw = factorial(q - row)
        fact_cl = factorial(q - col)
        return h^idx / (idx * fact_rw * fact_cl)
    end
    @fastmath function Q!(Q::AbstractMatrix, h::Real, σ²::Real=1.0)
        val = one(h)
        @simd for col in 0:q
            @simd for row in col:q
                val = _transdiff_ibm_element(row, col, h) * σ²
                @simd for i in 0:d-1
                    @inbounds Q[1 + col*d + i,1 + row*d + i] = val
                    @inbounds Q[1 + row*d + i,1 + col*d + i] = val
                end
            end
        end
    end

    return A!, Q!
end
