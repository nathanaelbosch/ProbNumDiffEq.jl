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

Implemented in Joseph Form to retain the `PSDMatrix` covariances:
```math
\\begin{aligned}
K &= Σ^P H^T S^{-1}, \\\\
μ^F &= μ + K (0 - \\hat{z}), \\\\
\\sqrt{Σ}^F &= (I - KH) \\sqrt(Σ),
\\end{aligned}
```
where ``\\sqrt{M}`` denotes the left square-root of a matrix M, i.e. ``M = \\sqrt{M} \\sqrt{M}^T``.

To prevent allocations, write into caches `K_cache` and `M_cache`, both of size `D × D`,
and `S_cache` of same type as `measurement.Σ`.

See also: [`update`](@ref).
"""
function update!(
    x_out::SRGaussian,
    x_pred::SRGaussian,
    measurement::Gaussian,
    H::AbstractMatrix,
    K1_cache::AbstractMatrix,
    K2_cache::AbstractMatrix,
    M_cache::AbstractMatrix,
    C_dxd::AbstractMatrix,
)
    z, S = measurement.μ, measurement.Σ
    m_p, P_p = x_pred.μ, x_pred.Σ
    @assert P_p isa PSDMatrix || P_p isa Matrix
    # The following is not ideal as `iszero` allocates
    # But, it is necessary to make the classic solver init stable
    if (P_p isa PSDMatrix && iszero(P_p.R)) || (P_p isa Matrix && iszero(P_p))
        copy!(x_out, x_pred)
        return x_out
    end

    D = size(m_p, 1)

    # K = P_p * H' / S
    _S = if S isa PSDMatrix
        _matmul!(C_dxd, S.R', S.R)
    else
        copy!(C_dxd, S)
    end

    K = if P_p isa PSDMatrix
        _matmul!(K1_cache, P_p.R, H')
        _matmul!(K2_cache, P_p.R', K1_cache)
    else
        _matmul!(K2_cache, P_p, H')
    end

    rdiv!(K, length(_S) == 1 ? _S[1] : cholesky!(_S))

    # x_out.μ .= m_p .+ K * (0 .- z)
    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)

    # M_cache .= I(D) .- mul!(M_cache, K, H)
    _matmul!(M_cache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        M_cache[i, i] += 1
    end

    fast_X_A_Xt!(x_out.Σ, P_p, M_cache)

    return x_out
end

# Kronecker version
function update!(
    x_out::SRGaussian{T,<:IsometricKroneckerProduct},
    x_pred::SRGaussian{T,<:IsometricKroneckerProduct},
    measurement::SRGaussian{T,<:IsometricKroneckerProduct},
    H::IsometricKroneckerProduct,
    K1_cache::IsometricKroneckerProduct,
    K2_cache::IsometricKroneckerProduct,
    M_cache::IsometricKroneckerProduct,
    C_dxd::IsometricKroneckerProduct,
) where {T}
    D = length(x_out.μ)  # full_state_dim
    d = H.ldim           # ode_dimension_dim
    Q = D ÷ d            # n_derivatives_dim
    _x_out = Gaussian(reshape_no_alloc(x_out.μ, Q, d), PSDMatrix(x_out.Σ.R.B))
    _x_pred = Gaussian(reshape_no_alloc(x_pred.μ, Q, d), PSDMatrix(x_pred.Σ.R.B))
    _measurement = Gaussian(
        reshape_no_alloc(measurement.μ, 1, d), PSDMatrix(measurement.Σ.R.B))
    _H = H.B
    _K1_cache = K1_cache.B
    _K2_cache = K2_cache.B
    _M_cache = M_cache.B
    _C_dxd = C_dxd.B

    return update!(
        _x_out, _x_pred, _measurement, _H, _K1_cache, _K2_cache, _M_cache, _C_dxd)
end
