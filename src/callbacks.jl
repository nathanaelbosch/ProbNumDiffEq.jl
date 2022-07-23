function manifoldupdate!(cache, residualf; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15)
    m, C = cache.x

    # Create some caches
    @unpack SolProj, tmp, H, x_tmp = cache
    D = cache.d * (cache.q + 1)
    z_tmp = residualf(mul!(tmp, SolProj, m))
    result = DiffResults.JacobianResult(z_tmp, tmp)
    d = length(z_tmp)
    H = H[1:d, :]
    _K1, _K2 = cache.C_DxD[:, 1:d], cache.C_2DxD[1:D, 1:d]

    S = PSDMatrix(C.R[:, 1:d])
    m_tmp, C_tmp = x_tmp

    m_i = copy(m)
    local m_i_new, C_i_new
    for i in 1:maxiters
        u_i = mul!(tmp, SolProj, m_i)

        ForwardDiff.jacobian!(result, residualf, u_i)
        z = DiffResults.value(result)
        J = DiffResults.jacobian(result)

        mul!(H, J, SolProj)
        X_A_Xt!(S, C, H)

        # m_i_new, C_i_new = update(x, Gaussian(z .+ (H * (m - m_i)), S), H)
        K = _matmul!(_K2, C.R', _matmul!(_K1, C.R, H' / S))
        m_tmp .= m_i .- m
        mul!(z_tmp, H, m_tmp)
        z_tmp .-= z
        mul!(m_tmp, K, z_tmp)
        m_i_new = m_tmp .+= m

        if (norm(m_i_new .- m_i) < ϵ₁ && norm(z) < ϵ₂) || (i == maxiters)
            C_i_new = X_A_Xt!(C_tmp, C, (I - K * H))
            break
        end
        m_i = m_i_new
    end

    copy!(cache.x, Gaussian(m_i_new, C_i_new))

    return nothing
end

"""
    ManifoldUpdate(residual::Function)

Update the state to satisfy a zero residual function via iterated extended Kalman filtering.

`ManifoldUpdate` returns a `SciMLBase.DiscreteCallback`, which, at each solver step,
performs an iterated extended Kalman filter update to keep the residual measurement to be
zero. Additional arguments and keyword arguments for the `DiscreteCallback` can be passed.

The residual function should be `residual(u::AbstractVector)::AbstractVector`, that is
_it should not be in-place_ (whereas DiffEqCallback.jl's `ManifoldProjection`) is.
If you encounter `SingularException`s, make sure that the residual function is such that
its Jacobian has full rank.

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
    affect!(integ) = begin
        manifoldupdate!(integ.cache, residual; maxiters=maxiters, ϵ₁=ϵ₁, ϵ₂=ϵ₂)
        mul!(view(integ.u, :), integ.cache.SolProj, integ.cache.x.μ)
    end
    return DiscreteCallback(condition, affect!, args...; kwargs...)
end
