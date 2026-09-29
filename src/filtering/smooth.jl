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

# Kronecker version
function smooth(
    x_curr::SRGaussian{T,<:IsometricKroneckerProduct},
    x_next_smoothed::SRGaussian{T,<:IsometricKroneckerProduct},
    Ah::IsometricKroneckerProduct,
    Qh::PSDMatrix{S,<:IsometricKroneckerProduct},
) where {T,S}
    d = Ah.rdim
    Q = length(x_curr.μ) ÷ d
    _x_curr = Gaussian(reshape(x_curr.μ, d, Q)', PSDMatrix(x_curr.Σ.R.B))
    _x_next = Gaussian(reshape(x_next_smoothed.μ, d, Q)', PSDMatrix(x_next_smoothed.Σ.R.B))
    _x, _G = smooth(_x_curr, _x_next, Ah.B, PSDMatrix(Qh.R.B))
    x = Gaussian(vec(permutedims(_x.μ)), PSDMatrix(IsometricKroneckerProduct(d, _x.Σ.R)))
    return x, IsometricKroneckerProduct(d, _G)
end

# BlocksOfDiagonals version
function smooth(
    x_curr::SRGaussian{T,<:BlocksOfDiagonals},
    x_next_smoothed::SRGaussian{T,<:BlocksOfDiagonals},
    Ah::BlocksOfDiagonals,
    Qh::PSDMatrix{S,<:BlocksOfDiagonals},
) where {T,S}
    d = nblocks(Ah)
    μ = similar(x_curr.μ)
    R_blocks, G_blocks = similar(blocks(Ah)), similar(blocks(Ah))
    @views for i in eachindex(blocks(Ah))
        _x, _G = smooth(
            Gaussian(x_curr.μ[i:d:end], PSDMatrix(x_curr.Σ.R.blocks[i])),
            Gaussian(x_next_smoothed.μ[i:d:end], PSDMatrix(x_next_smoothed.Σ.R.blocks[i])),
            Ah.blocks[i],
            PSDMatrix(Qh.R.blocks[i]),
        )
        μ[i:d:end] .= _x.μ
        R_blocks[i] = _x.Σ.R
        G_blocks[i] = _G
    end
    return Gaussian(μ, PSDMatrix(BlocksOfDiagonals(R_blocks))), BlocksOfDiagonals(G_blocks)
end
