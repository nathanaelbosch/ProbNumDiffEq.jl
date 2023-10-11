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
    # The following could be useful; but this never really happens and `iszero` allocates
    if (P_p isa PSDMatrix && iszero(P_p.R)) || (P_p isa Matrix && iszero(P_p))
        copy!(x_out, x_pred)
        return x_out
    end

    D = length(m_p)

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

    S_chol = if length(_S) == 1
        _S[1]
    else
        try
            cholesky!(_S)
        catch e
            if !(e isa PosDefException)
                throw(e)
            end
            @warn "Can't compute the update step with cholesky; using qr instead" _S
            @assert S isa PSDMatrix
            error()
            Cholesky(qr(S.R).R, :U, 0)
        end
    end
    rdiv!(K, S_chol)

    # x_out.μ .= m_p .+ K * (0 .- z)
    a, b = size(K)
    D, d = length(m_p), length(z)
    if a == D
        _matmul!(x_out.μ, K, z)
    else
        _matmul!(reshape_no_alloc(x_out.μ, a, d), K, reshape_no_alloc(z, 1, d))
    end
    x_out.μ .= m_p .- x_out.μ

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
    x_out::SRGaussian{T,<:Kronecker.KroneckerProduct},
    x_pred::SRGaussian{T,<:Kronecker.KroneckerProduct},
    measurement::SRGaussian{T,<:Kronecker.KroneckerProduct},
    H::Kronecker.KroneckerProduct,
    K1_cache::AbstractMatrix,
    K2_cache::AbstractMatrix,
    M_cache::AbstractMatrix,
    C_dxd::AbstractMatrix,
) where {T}
    _x_out = Gaussian(x_out.μ, PSDMatrix(x_out.Σ.R.B))
    _x_pred = Gaussian(x_pred.μ, PSDMatrix(x_pred.Σ.R.B))
    _measurement = Gaussian(measurement.μ, PSDMatrix(measurement.Σ.R.B))
    _H = H.B
    o = size(_measurement.Σ, 1)
    d = size(H.A, 1)
    _D = length(x_out.μ) ÷ d
    _K1_cache = view(K1_cache, 1:_D, 1:o)
    _K2_cache = view(K2_cache, 1:_D, 1:o)
    _M_cache = view(M_cache, 1:_D, 1:_D)
    _C_dxd = view(C_dxd, 1:o, 1:o)
    return update!(
        _x_out,
        _x_pred,
        _measurement,
        _H,
        _K1_cache,
        _K2_cache,
        _M_cache,
        _C_dxd,
    )
end
