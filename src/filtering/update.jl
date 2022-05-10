"""
    update(x, measurement, H)

Update step in Kalman filtering for linear dynamics models.

Given a Gaussian ``x = \\mathcal{N}(μ, Σ)``
and a measurement ``z = \\mathcal{N}(\\hat{z}, S)``, with ``S = H Σ H^T``,
compute
```math
\\begin{aligned}
K &= Σ^P H^T S^{-1}, \\\\
μ^F &= μ + K (0 - \\hat{z}), \\\\
Σ^F &= Σ - K S K^T,
\\end{aligned}
```
and return an updated state `\\mathcal{N}(μ^F, Σ^F)`.
Note that this assumes zero-measurements.
When called with `ProbNumDiffEq.SquarerootMatrix` type arguments it performs the update in
Joseph / square-root form.

For better performance, we recommend to use the non-allocating [`update!`](@ref).
"""
function update(x::Gaussian, measurement::Gaussian, H::AbstractMatrix)
    m, C = x
    z, S = measurement

    K = C * H' * inv(S)
    m_new = m - K * z
    C_new = C - K * S * K'

    return Gaussian(m_new, C_new)
end
function update(x::SRGaussian, measurement::Gaussian, H::AbstractMatrix)
    m, C = x
    z, S = measurement

    K = C * H' * inv(S)
    m_new = m - K * z
    C_new = X_A_Xt(C, (I - K * H))

    return Gaussian(m_new, C_new)
end

"""
    update!(x_out, x_pred, measurement, H, K_cache, M_cache, S_cache)

In-place and square-root implementation of [`update`](@ref)
which saves the result into `x_out`.

Implemented in Joseph Form; works best when called with `SquarerootMatrix` covariances.
To prevent allocations, write into caches `K_cache` and `M_cache`, both of size `D × D`,
and `S_cache` of same type as `measurement.Σ`.

See also: [`update`](@ref).
"""
function update!(
    x_out::Gaussian,
    x_pred::Gaussian,
    measurement::Gaussian,
    H::AbstractMatrix,
    K_cache::AbstractMatrix,
    M_cache::AbstractMatrix,
    S_cache::AbstractMatrix,
)
    z, S = measurement.μ, copy!(S_cache, measurement.Σ)
    m_p, P_p = x_pred.μ, x_pred.Σ
    D = length(m_p)

    # K = P_p * H' / S
    if S isa SRMatrix
        S_chol = Cholesky(custom_qr!(S.R).R, :U, 0)
    else
        S_chol = cholesky(S)
    end
    K = _matmul!(K_cache, Matrix(P_p), H')
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
