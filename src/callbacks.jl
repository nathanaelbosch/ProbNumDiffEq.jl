function manifoldupdate!(integ, residualf; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15)
    @unpack x, SolProj = integ.cache
    f(m) = residualf(SolProj * m)
    return copy!(x, manifoldupdate(x, f; maxiters=maxiters, ϵ₁=ϵ₁, ϵ₂=ϵ₂))
end

function manifoldupdate(x, f; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15)
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
    return Gaussian(m_i_new, C_i_new)
end

function ManifoldUpdate(residualf, args...; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15, kwargs...)
    condition(u, t, integ) = true
    affect!(integ) = manifoldupdate!(integ, residualf; maxiters=maxiters, ϵ₁=ϵ₁, ϵ₂=ϵ₂)
    return DiscreteCallback(condition, affect!, args...; kwargs...)
end
