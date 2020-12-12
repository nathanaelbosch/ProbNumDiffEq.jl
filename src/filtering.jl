"""Gaussian filtering and smoothing"""


"""
    predict!(x_out, x_curr, Ah, Qh)

PREDICT step in Kalman filtering for linear dynamics models.
In-place implementation of [`predict`](@ref), saving the result in `x_out`.

```math
m_{n+1}^P = A(h)*m_n
P_{n+1}^P = A(h)*P_n*A(h) + Q(h)
```

See also: [`predict`](@ref)
"""
function predict!(x_out::Gaussian, x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    mul!(x_out.μ, Ah, x_curr.μ)
    out_cov = X_A_Xt(x_curr.Σ, Ah) + Qh
    copy!(x_out.Σ, out_cov)
    return nothing
end
function predict!(x_out::PSDGaussian, x_curr::PSDGaussian, Ah::AbstractMatrix, Qh::PSDMatrix)
    mul!(x_out.μ, Ah, x_curr.μ)
    _, R = qr([Ah*x_curr.Σ.L Qh.L]')
    out_cov = PSDMatrix(LowerTriangular(collect(R')))
    copy!(x_out.Σ, out_cov)
    return nothing
end

"""
    predict(x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)

See also: [`predict!`](@ref)
"""
function predict(x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    x_out = copy(x_curr)
    predict!(x_out, x_curr, Ah, Qh)
    return x_out
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
function update!(x_out::Gaussian, x_pred::Gaussian, measurement::Gaussian,
                 H::AbstractMatrix, R=0)
    @assert iszero(R)
    z, S = measurement.μ, measurement.Σ
    m_p, P_p = x_pred.μ, x_pred.Σ

    S_inv = inv(S)
    K = P_p * H' * S_inv

    x_out.μ .= m_p .+ K * (0 .- z)
    copy!(x_out.Σ, X_A_Xt(P_p, (I-K*H)))
    return x_out
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
function smooth(x_curr::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    x_pred = predict(x_curr, Ah, Qh)

    P_p = x_pred.Σ
    P_p_inv = inv(P_p)

    G = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - x_pred.μ)

    smoothed_cov = (
        X_A_Xt(x_curr.Σ, (I - G*Ah))
        + X_A_Xt(Qh, G)
        + X_A_Xt(x_next_smoothed.Σ, G)
    )

    assert_nonnegative_diagonal(smoothed_cov)

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
