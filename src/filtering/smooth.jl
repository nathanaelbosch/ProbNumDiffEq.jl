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

function smooth!(x_curr, x_next, Ah, Qh, integ, diffusion=1)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    @unpack d, q = integ.cache
    @unpack x_pred = integ.cache
    @unpack C1, G1, G2, C2 = integ.cache

    # Prediction: t -> t+1
    predict_mean!(x_pred, x_curr, Ah)
    predict_cov!(x_pred, x_curr, Ah, Qh, C1, diffusion)

    # Smoothing
    # G = x_curr.Σ * Ah' * P_p_inv
    P_p_chol = Cholesky(x_pred.Σ.squareroot, :L, 0)
    G = rdiv!(_matmul!(G1, x_curr.Σ.mat, Ah'), P_p_chol)

    x_curr.μ .+= G * (x_next.μ .- x_pred.μ)

    # Joseph-Form:
    M, L = C2.mat, C2.squareroot
    D = length(x_pred.μ)

    _matmul!(G2, G, Ah)
    copy!(view(L, 1:D, 1:D), x_curr.Σ.squareroot)
    _matmul!(view(L, 1:D, 1:D), G2, x_curr.Σ.squareroot, -1.0, 1.0)

    _matmul!(view(L, 1:D, D+1:2D), _matmul!(G2, G, sqrt.(diffusion)), Qh.squareroot)
    _matmul!(view(L, 1:D, 2D+1:3D), G, x_next.Σ.squareroot)

    # _matmul!(M, L, L')
    # chol = cholesky!(Symmetric(M), check=false)
    succ = false && issuccess(chol)

    QL =
        succ ? Matrix(chol.U)' :
        eltype(L) <: Union{Float16,Float32,Float64} ? lq!(L).L : qr(L').R'
    copy!(x_curr.Σ.squareroot, QL)
    _matmul!(x_curr.Σ.mat, QL, QL')

    return nothing
end
