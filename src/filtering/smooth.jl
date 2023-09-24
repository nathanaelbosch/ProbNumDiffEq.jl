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

For better performance, we recommend to use the non-allocating [`smooth!`](@ref).
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

    G = Matrix(x_curr.Σ) * Ah' / x_pred.Σ

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

"""
    smooth!(x_curr, x_next, Ah, Qh, cache, diffusion=1)

In-place and square-root implementation of [`smooth`](@ref) which overwrites `x_curr`.

Implemented in Joseph form to preserve square-root structure.
It requires access to the solvers `cache`
to prevent allocations.

See also: [`smooth`](@ref).
"""
function smooth!(
    x_curr::SRGaussian,
    x_next::SRGaussian,
    Ah::AbstractMatrix,
    Qh::PSDMatrix,
    cache,
    diffusion::Union{Number,Diagonal}=1,
)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    @unpack x_pred = cache
    @unpack G1, C_DxD, C_2DxD, C_3DxD = cache
    D = length(x_curr.μ)
    _D = size(C_DxD, 1)

    # Prediction: t -> t+1
    predict_mean!(x_pred, x_curr, Ah)
    predict_cov!(x_pred, x_curr, Ah, Qh, C_DxD, C_2DxD, diffusion)

    # Smoothing
    # G = x_curr.Σ * Ah' * P_p_inv
    P_p_chol = Cholesky(x_pred.Σ.R, :U, 0)
    G = rdiv!(_matmul!(G1, x_curr.Σ.R', _matmul!(C_DxD, x_curr.Σ.R, Ah')), P_p_chol)

    # x_curr.μ .+= G * (x_next.μ .- x_pred.μ) # less allocations:
    x_pred.μ .-= x_next.μ
    a = D ÷ _D
    _matmul!(reshape_no_alloc(x_curr.μ, _D, a), G, reshape_no_alloc(x_pred.μ, _D, a), -1, 1)

    # Joseph-Form:
    R = C_3DxD

    G2 = _matmul!(C_DxD, G, Ah)
    copy!(view(R, 1:_D, 1:_D), x_curr.Σ.R)
    _matmul!(view(R, 1:_D, 1:_D), x_curr.Σ.R, G2', -1.0, 1.0)

    _matmul!(view(R, _D+1:2_D, 1:_D), Qh.R, _matmul!(G2, G, sqrt.(diffusion))')
    _matmul!(view(R, 2_D+1:3_D, 1:_D), x_next.Σ.R, G')

    Q_R = triangularize!(R, cachemat=C_DxD)
    copy!(x_curr.Σ.R, Q_R)

    return nothing
end


function smooth!(
    x_curr::SRGaussian{T,<:Kronecker.KroneckerProduct},
    x_next::SRGaussian{T,<:Kronecker.KroneckerProduct},
    Ah::Kronecker.KroneckerProduct,
    Qh::PSDMatrix{S,<:Kronecker.KroneckerProduct},
    cache,
    diffusion::Union{Number,Diagonal}=1,
) where {T,S}
    _x_curr = Gaussian(x_curr.μ, PSDMatrix(x_curr.Σ.R.B))
    _x_next = Gaussian(x_next.μ, PSDMatrix(x_next.Σ.R.B))
    _Ah = Ah.B
    _Qh = PSDMatrix(Qh.R.B)
    _D = size(_Qh, 1)
    _cache = (
        G1 = view(cache.G1, 1:_D, 1:_D),
        C_DxD = view(cache.C_DxD, 1:_D, 1:_D),
        C_2DxD = view(cache.C_2DxD, 1:2*_D, 1:_D),
        C_3DxD = view(cache.C_3DxD, 1:3*_D, 1:_D),
        x_pred = Gaussian(cache.x_pred.μ, PSDMatrix(cache.x_pred.Σ.R.B))
    )
    smooth!(
        _x_curr,
        _x_next,
        _Ah,
        _Qh,
        _cache,
        diffusion,
    )
end
