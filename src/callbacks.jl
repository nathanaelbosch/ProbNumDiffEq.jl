function ManifoldUpdate(residualf; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15)
    condition(u, t, integ) = true

    function affect!(integ)
        @unpack u = integ
        @unpack x, SolProj = integ.cache

        f(m) = residualf(SolProj * m)

        m, C = x
        m_i = copy(m)

        local m_i_new, C_i_new
        for i in 1:maxiters
            z = f(m_i)
            J = ForwardDiff.jacobian(f, m_i)
            S = X_A_Xt(C, J)

            m_i_new, C_i_new = update(x, Gaussian(z .+ (J * (m - m_i)), S), J)

            if norm(m_i_new .- m_i) < ϵ₁ && norm(z) < ϵ₂
                break
            end
            m_i = m_i_new
        end
        copy!(x, Gaussian(m_i_new, C_i_new))

        return nothing
    end
    return DiscreteCallback(condition, affect!)
end
