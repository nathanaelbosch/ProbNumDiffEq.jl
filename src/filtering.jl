"""Gaussian filtering and smoothing"""

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
    R,
    K1::AbstractMatrix,
    K2::AbstractMatrix,
    M_cache::AbstractMatrix,
    m_tmp,
)
    @assert iszero(R)
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

"""
    smooth(x_curr::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)

SMOOTH step of the (extended) Kalman smoother, or (extended) Rauch-Tung-Striebel smoother.
It is implemented in Joseph Form:
```math
m_{n+1}^P = A(h)*m_n
P_{n+1}^P = A(h)*P_n*A(h) + Q(h)

G = P_n * A(h)^T * (P_{n+1}^P)^{-1}
m_n^S = m_n + G * (m_{n+1}^S - m_{n+1}^P)
P_n^S = (I - G*A(h)) P_n (I - G*A(h))^T + G * Q(h) * G + G * P_{n+1}^S * G
```
"""
function smooth(
    x_curr::Gaussian,
    x_next_smoothed::Gaussian,
    Ah::AbstractMatrix,
    Qh::AbstractMatrix,
)
    x_pred = predict(x_curr, Ah, Qh)

    P_p = x_pred.Σ
    P_p_inv = inv(P_p)

    G = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - x_pred.μ)
    smoothed_cov =
        (X_A_Xt(x_curr.Σ, (I - G * Ah)) + X_A_Xt(Qh, G) + X_A_Xt(x_next_smoothed.Σ, G))
    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
function smooth(
    x_curr::SRGaussian,
    x_next_smoothed::SRGaussian,
    Ah::AbstractMatrix,
    Qh::SRMatrix,
)
    x_pred = predict(x_curr, Ah, Qh)

    P_p = x_pred.Σ
    P_p_inv = inv(P_p)

    G = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - x_pred.μ)

    _R = [
        x_curr.Σ.squareroot' * (I - G * Ah)'
        Qh.squareroot' * G'
        x_next_smoothed.Σ.squareroot' * G'
    ]
    _, P_s_R = qr(_R)
    smoothed_cov = SRMatrix(P_s_R')

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
