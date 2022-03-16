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

"""
    ManifoldUpdate(residual::Function)

Update the state to satisfy a zero residual function via iterated extended Kalman filtering.

`ManifoldUpdate` returns a `SciMLBase.DiscreteCallback`, which, at each solver step,
performs an iterated extended Kalman filter update to keep the residual measurement to be
zero. Additional arguments and keyword arguments for the `DiscreteCallback` can be passed.

# Additional keyword arguments
- `maxiters::Int`: Maximum number of IEKF iterations.
  Setting this to 1 results in a single standard EKF update.
"""
function ManifoldUpdate(
    residual::Function,
    args...;
    maxiters=100,
    ϵ₁=1e-25,
    ϵ₂=1e-15,
    kwargs...,
)
    condition(u, t, integ) = true
    affect!(integ) = manifoldupdate!(integ, residual; maxiters=maxiters, ϵ₁=ϵ₁, ϵ₂=ϵ₂)
    return DiscreteCallback(condition, affect!, args...; kwargs...)
end
