########################################################################################
# Integrated Brownian Motion
########################################################################################
"""
    ibm(d::Integer, q::Integer, elType=typeof(1.0))

Generate the discrete dynamics for a q-IBM model. INCLUDES AUTOMATIC PRECONDITIONING!

Careful: Dimensions are ordered differently than in `probnum`!"""
function ibm_old(d::Integer, q::Integer, elType=typeof(1.0))

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
    A!(A_base, one(elType))
    @assert istriu(A_base)
    A_base = UpperTriangular(A_base)

    @fastmath function _transdiff_ibm_element(row::Int, col::Int, h::Real)
        idx = 2 * q + 1 - row - col
        if q > 10 || h isa BigFloat
            fact_rw = factorial(big(q - row))
            fact_cl = factorial(big(q - col))
            return h / (idx * fact_rw * fact_cl)
        else
            fact_rw = factorial(q - row)
            fact_cl = factorial(q - col)
            return h / (idx * fact_rw * fact_cl)
        end
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

    Q!(Q_base, one(elType))
    QL = cholesky(Q_base).L
    Q_psd = SRMatrix(QL)

    return A_base, Q_psd
end


function ibm(d::Integer, q::Integer, elType=typeof(1.0))
    D = d*(q+1)

    # Make A
    A_breve = zeros(elType, q+1, q+1)
    @simd ivdep for j in 1:q+1
        @simd ivdep for i in 1:j
            @inbounds A_breve[i, j] = binomial(q-i+1, q-j+1)
        end
    end
    A = kron(A_breve, I(d))
    @assert istriu(A)
    A = UpperTriangular(A)

    # Make Q
    Q_breve = zeros(elType, q+1, q+1)
    @fastmath _transdiff_ibm_element(row::Int, col::Int) =
        one(elType) / (2 * q + 1 - row - col)
    @simd for col in 0:q
        @simd for row in col:q
            val = _transdiff_ibm_element(row, col)
            @inbounds Q_breve[1 + col, 1 + row] = val
            @inbounds Q_breve[1 + row, 1 + col] = val
        end
    end
    QL_breve = cholesky(Q_breve).L
    QL = kron(QL_breve, I(d))
    Q = SRMatrix(QL)

    return A, Q
end


"""Same as above, but without the automatic preconditioning"""
function vanilla_ibm(d::Integer, q::Integer)

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
