########################################################################################
# Integrated Brownian Motion
########################################################################################
"""
    ibm(d::Integer, q::Integer, elType=typeof(1.0))

Generate the discrete dynamics for a q-IBM model.

The returned matrices `A::AbstractMatrix` and `Q::ProbNumDiffEq.SquarerootMatrix` should be
used in combination with the preconditioners (see `./src/preconditioning.jl`).
"""
function ibm(d::Integer, q::Integer, ::Type{elType}=typeof(1.0)) where {elType}
    # Make A
    A_breve = zeros(elType, q + 1, q + 1)
    @simd ivdep for j in 1:q+1
        @simd ivdep for i in 1:j
            @inbounds A_breve[i, j] = binomial(q - i + 1, q - j + 1)
        end
    end
    A = kron(I(d), A_breve)
    @assert istriu(A)
    # A = UpperTriangular(A)

    # Make Q
    Q_breve = zeros(elType, q + 1, q + 1)
    @fastmath _transdiff_ibm_element(row::Int, col::Int) =
        one(elType) / (2 * q + 1 - row - col)
    @simd ivdep for col in 0:q
        @simd ivdep for row in 0:q
            val = _transdiff_ibm_element(row, col)
            @inbounds Q_breve[1+row, 1+col] = val
        end
    end
    QL_breve = cholesky(Q_breve).L
    Q = SRMatrix(kron(I(d), QL_breve))

    return A, Q
end

"""
    vanilla_ibm(d::Integer, q::Integer)

**This function serves only for tests and is not used anywhere in the main package!**
"""
function vanilla_ibm(d::Integer, q::Integer)
    @fastmath function A!(A::AbstractMatrix, h::Real)
        # Assumes that A comes from a previous computation => zeros and one-diag
        val = one(h)
        for i in 1:q
            val = val * h / i
            for k in 0:d-1
                for j in 1:q+1-i
                    @inbounds A[j+k*(q+1), j+k*(q+1)+i] = val
                end
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
                    @inbounds Q[1+col+i*(q+1), 1+row+i*(q+1)] = val
                    @inbounds Q[1+row+i*(q+1), 1+col+i*(q+1)] = val
                end
            end
        end
    end

    return A!, Q!
end
