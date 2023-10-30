"""
    smooth(x_curr, x_next_smoothed, A, Q)

Update step of the Kalman smoother, aka. Rauch-Tung-Striebel smoother,
for linear dynamics models.

Given Gaussians
``x_n = \\mathcal{N}(μ_{n}, Σ_{n})`` and
``x_{n+1} = \\mathcal{N}(μ_{n+1}^S, Σ_{n+1}^S)``,
compute
```math
\\begin{aligned}
μ_{n+1}^P &= A μ_n^F, \\\\
P_{n+1}^P &= A Σ_n^F A + Q, \\\\
G &= Σ_n^S A^T (Σ_{n+1}^P)^{-1}, \\\\
μ_n^S &= μ_n^F + G (μ_{n+1}^S - μ_{n+1}^P), \\\\
Σ_n^S &= (I - G A) Σ_n^F (I - G A)^T + G Q G^T + G Σ_{n+1}^S G^T,
\\end{aligned}
```
and return a smoothed state `\\mathcal{N}(μ_n^S, Σ_n^S)`.
When called with `ProbNumDiffEq.SquarerootMatrix` type arguments it performs the update in
Joseph / square-root form.
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
    Qh::PSDMatrix,
)
    x_pred = predict(x_curr, Ah, Qh)

    G = x_curr.Σ.R' * x_curr.Σ.R * Ah' / x_pred.Σ

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - x_pred.μ)

    _R = [
        x_curr.Σ.R * (I - G * Ah)'
        Qh.R * G'
        x_next_smoothed.Σ.R * G'
    ]
    P_s_R = qr(_R).R
    smoothed_cov = PSDMatrix(P_s_R)

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
