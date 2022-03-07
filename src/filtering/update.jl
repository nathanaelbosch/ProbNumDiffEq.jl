"""
    update(x, measurement, H)

UPDATE step in Kalman filtering for linear dynamics models:
```math
K = P_{n+1}^P * H^T * S^{-1}
m_{n+1} = m_{n+1}^P + K * (0 - z)
P_{n+1} = P_{n+1}^P - K*S*K^T
```

This function provides a very simple UPDATE implementation.
In the solvers, we recommend to use the non-allocating [`update!`](@ref).
"""
function update(x::Gaussian, measurement::Gaussian, H::AbstractMatrix)
    m, C = x
    z, S = measurement

    K = C * H' * inv(S)
    m_new = m - K * z
    C_new = C - K * S * K'

    return Gaussian(m_new, C_new)
end
"""UPDATE step in Joseph-form, with square-root matrix inputs"""
function update(x::SRGaussian, measurement::Gaussian, H::AbstractMatrix)
    m, C = x
    z, S = measurement

    K = C * H' * inv(S)
    m_new = m - K * z
    C_new = X_A_Xt(C, (I - K * H))

    return Gaussian(m_new, C_new)
end

"""
    update!(x_out, x_pred, measurement, H, R=0)

UPDATE step in Kalman filtering for linear dynamics models, given a measurement `Z=N(z, S)`.
In-place implementation of [`update`](@ref), saving the result in `x_out`.

```math
K = P_{n+1}^P * H^T * S^{-1}
m_{n+1} = m_{n+1}^P + K * (0 - z)
P_{n+1} = P_{n+1}^P - K*S*K^T
```

Implemented in Joseph Form.

See also: [`predict`](@ref)
"""
function update!(
    x_out::Gaussian,
    x_pred::Gaussian,
    measurement::Gaussian,
    H::AbstractMatrix,
    K_cache::AbstractMatrix,
    M_cache::AbstractMatrix,
    S_cache::AbstractMatrix,
)
    z, S = measurement.μ, copy!(S_cache, measurement.Σ.mat)
    m_p, P_p = x_pred.μ, x_pred.Σ
    D = length(m_p)

    # K = P_p * H' / S
    S_chol = cholesky!(S)
    K = _matmul!(K_cache, Matrix(P_p), H')
    rdiv!(K, S_chol)

    # x_out.μ .= m_p .+ K * (0 .- z)
    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)

    # M_cache .= I(D) .- mul!(M_cache, K, H)
    _matmul!(M_cache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        M_cache[i, i] += 1
    end

    X_A_Xt!(x_out.Σ, P_p, M_cache)

    return x_out
end
