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
    K1::AbstractMatrix,
    K2::AbstractMatrix,
    M_cache::AbstractMatrix,
    m_tmp,
)

    z, S = measurement.μ, copy!(m_tmp.Σ, measurement.Σ)
    m_p, P_p = x_pred.μ, x_pred.Σ
    D = length(m_p)

    # K = P_p * H' / S
    S_chol = cholesky!(S)
    K = _matmul!(K1, Matrix(P_p), H')
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
function update!(
    x_out::Gaussian,
    x_pred::Gaussian,
    measurement::Gaussian,
    H::AbstractMatrix,
    R,
)
    D = length(x_out.μ)
    d = length(measurement.μ)
    K1 = zeros(D, d)
    K2 = zeros(D, d)
    M_cache = zeros(D, D)
    m_tmp = copy(measurement)
    return update!(x_out, x_pred, measurement, H, R, K1, K2, M_cache, m_tmp)
end
"""
    update(x_pred, measurement, H, R=0)

See also: [`update!`](@ref)
"""
function update(x_pred::Gaussian, measurement::Gaussian, H::AbstractMatrix, R=0)
    @assert iszero(R)
    x_out = copy(x_pred)
    update!(x_out, x_pred, measurement, H, R)
    return x_out
end
