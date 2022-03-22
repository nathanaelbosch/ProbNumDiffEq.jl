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

"""
    smooth!(x_curr, x_next, Ah, Qh, cache, diffusion=1)

In-place and square-root implementation of [`smooth`](@ref) which overwrites `x_curr`.

Implemented in Joseph form to preserve square-root structure.
It requires access to the solvers `GaussianODEFilterCache` (passed as `cache`)
to prevent allocations.

See also: [`smooth`](@ref).
"""
function smooth!(
    x_curr::SRGaussian,
    x_next::SRGaussian,
    Ah::AbstractMatrix,
    Qh::SRMatrix,
    cache::GaussianODEFilterCache,
    diffusion::Union{Number,Diagonal}=1,
)
    D = length(x_curr.μ)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    @unpack d, q = cache
    @unpack x_pred = cache
    @unpack C1, G1, G2, C2 = cache

    # Prediction: t -> t+1
    predict_mean!(x_pred, x_curr, Ah)
    predict_cov!(x_pred, x_curr, Ah, Qh, C1, diffusion)

    # Smoothing
    # G = x_curr.Σ * Ah' * P_p_inv
    P_p_chol = Cholesky(x_pred.Σ.squareroot, :L, 0)
    G = rdiv!(_matmul!(G1, x_curr.Σ.mat, Ah'), P_p_chol)

    # x_curr.μ .+= G * (x_next.μ .- x_pred.μ) # less allocations:
    x_pred.μ .-= x_next.μ
    _matmul!(x_curr.μ, G, x_pred.μ, -1, 1)

    # Joseph-Form:
    M, L = C2.mat, C2.squareroot

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
