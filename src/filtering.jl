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
    predict_mean!(x_out, x_curr, Ah, Qh)
    predict_cov!(x_out, x_curr, Ah, Qh)
    return x_out
end
function predict_mean!(x_out::Gaussian, x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    mul!(x_out.μ, Ah, x_curr.μ)
    return x_out.μ
end
function predict_cov!(x_out::Gaussian, x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    out_cov = X_A_Xt(x_curr.Σ, Ah) + Qh
    copy!(x_out.Σ, out_cov)
    return x_out.Σ
end

# SRMatrix Version of this - But, before using QR try it with Cholesky!
function predict_cov!(x_out::SRGaussian, x_curr::SRGaussian, Ah::AbstractMatrix, Qh::SRMatrix)
    _L = [Ah*x_curr.Σ.squareroot Qh.squareroot]
    out_cov = Symmetric(_L*_L')
    chol = cholesky!(out_cov, check=false)

    if issuccess(chol)
        PpL = chol.L
        copy!(x_out.Σ, SRMatrix(PpL))
        return x_out.Σ
    else
        _, R = qr(_L')
        out_cov = SRMatrix(LowerTriangular(collect(R')))
        copy!(x_out.Σ, out_cov)
        return x_out.Σ
    end
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
# Version with scalar measurement and vector measurement matrix
function update!(x_out::Gaussian, x_pred::Gaussian, measurement::Gaussian,
                 H::AbstractVector, R=0)
    @assert iszero(R)
    z, S = measurement.μ, measurement.Σ
    @assert isreal(z) && isreal(S)

    m_p, P_p = x_pred.μ, x_pred.Σ
    K = P_p * H / S

    x_out.μ .= m_p .+ K * (0 .- z)
    copy!(x_out.Σ, X_A_Xt(P_p, (I-K*H')))
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




function iekf_update(x::Gaussian, h::Function, z=0, R=0;
                     maxiters=100)
    # maxiters=1 basically makes it an EKF

    ϵ₁ = 1e-25
    ϵ₂ = 1e-15

    i=0
    μ_i = copy(x.μ)
    local K_i, S_i, H_i, h_i
    # result = DiffResults.GradientResult(μ_i)
    for i in 1:maxiters

        # Measure
        # result = ForwardDiff.gradient!(result, h, μ_i)
        # h_i = DiffResults.value(result)
        # H_i = DiffResults.gradient(result)
        # H_i = reshape(H_i, (1, length(H_i)))
        h_i = h(μ_i)
        H_i = ForwardDiff.jacobian(h, μ_i)
        # H_i = reshape(H_i, (1, length(H_i)))

        S_i_L = H_i*x.Σ.squareroot
        S_i = S_i_L * S_i_L'
        K_i = x.Σ * H_i' * inv(S_i)
        μ_i_new = x.μ .+ K_i * (z .- h_i .- (H_i * (x.μ - μ_i)))

        # @info "iekf" i h_i norm(μ_i_new .- μ_i) μ_i μ_i_new
        # μ_i μ_i_new

        if norm(μ_i_new .- μ_i) < ϵ₁ && norm(h_i) < ϵ₂
            μ_i = μ_i_new
            break
        end

        μ_i = μ_i_new
    end
    P_i = X_A_Xt(x.Σ, (I-K_i*H_i))
    out = Gaussian(μ_i, P_i)

    # if abs(h_i) > ϵ₂
    #     @error "Quantity too large, but iteration done!" K_i S_i H_i h_i μ_i P_i
    #     error()
    # end

    return out
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
    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
function smooth(x_curr::SRGaussian, x_next_smoothed::SRGaussian, Ah::AbstractMatrix, Qh::SRMatrix)
    x_pred = predict(x_curr, Ah, Qh)

    P_p = x_pred.Σ
    P_p_inv = inv(P_p)

    G = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - x_pred.μ)

    _R = [x_curr.Σ.squareroot' * (I - G*Ah)'
          Qh.squareroot' * G'
          x_next_smoothed.Σ.squareroot' * G']
    _, P_s_R = qr(_R)
    smoothed_cov = SRMatrix(P_s_R')

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
