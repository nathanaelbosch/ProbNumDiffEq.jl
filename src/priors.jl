########################################################################################
# Integrated Brownian Motion
########################################################################################
"""Generate the discrete dynamics for a q-IBM model

Careful: Dimensions are ordered differently than in `probnum`!"""
function ibm(q::Integer, d::Integer; σ::Integer=1)
    F̃ = diagm(1 => ones(q))
    I_d = diagm(0 => ones(d))
    F = kron(F̃, I_d)  # In probnum the order is inverted

    # L̃ = zeros(q+1)
    # L̃[end] = σ^2
    # I_d = diagm(0 => ones(d))
    # L = kron(L̃, I_d)'  # In probnum the order is inverted

    A(h) = UpperTriangular(exp(F*(h)))

    function Q(h)
        function _transdiff_ibm_element(row, col)
            idx = 2 * q + 1 - row - col
            fact_rw = factorial(q - row)
            fact_cl = factorial(q - col)

            return σ ^ 2 * (h ^ idx) / (idx * fact_rw * fact_cl)
        end

        qh_1d = [_transdiff_ibm_element(row, col) for col in 0:q, row in 0:q]
        I_d = diagm(0 => ones(d))
        return kron(qh_1d, I_d)
    end

    return (A=A, Q=Q, q=q, d=d)
end
