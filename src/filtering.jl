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
    # TODO: This can be done more efficiently
    out_cov = X_A_Xt(x_curr.Σ, Ah)
    if !iszero(Qh)
        out_cov = out_cov + Qh
    end
    copy!(x_out.Σ, out_cov)
    return nothing
end
"""
    predict(x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)

PREDICT step in Kalman filtering for linear dynamics models.

```math
m_{n+1}^P = A(h)*m_n
P_{n+1}^P = A(h)*P_n*A(h) + Q(h)
```

See also: [`predict!`](@ref)
"""
function predict(x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    x_out = copy(x_curr)
    predict!(x_out, x_curr, Ah, Qh)
    return x_out
end


"""Vanilla UPDATE, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against
"""
function update(x_pred::Gaussian, h::AbstractVector, H::AbstractMatrix, R::AbstractMatrix)
    m_p, P_p = x_pred.μ, x_pred.Σ

    v = 0 .- h
    S = Symmetric(H * P_p * H' .+ R)
    S_inv = inv(S)
    K = P_p * H' * S_inv

    filt_mean = m_p .+ P_p * H' * (S\v)
    filt_cov = X_A_Xt(P_p, (I-K*H))
    @assert iszero(R)

    assert_nonnegative_diagonal(filt_cov)

    return Gaussian(filt_mean, filt_cov)
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
