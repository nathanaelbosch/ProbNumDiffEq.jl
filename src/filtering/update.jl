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
    m, C = mean(x), cov(x)
    z, S = mean(measurement), cov(measurement)

    K = C * H' * inv(S)
    m_new = m - K * z
    C_new = C - K * S * K'

    return Gaussian(m_new, C_new)
end
function update(x::SRGaussian, measurement::Gaussian, H::AbstractMatrix)
    m, C = mean(x), cov(x)
    z, S = mean(measurement), cov(measurement)

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
    C_d::AbstractArray;
    R::Union{Nothing,PSDMatrix}=nothing,
)
    z, S = measurement.μ, measurement.Σ
    m_p, P_p = x_pred.μ, x_pred.Σ
    @assert P_p isa PSDMatrix || P_p isa Matrix
    # The following is not ideal as `iszero` allocates
    # But, it is necessary to make the classic solver init stable
    if (P_p isa PSDMatrix && iszero(P_p.R)) || (P_p isa Matrix && iszero(P_p))
        copy!(x_out, x_pred)
        if iszero(z)
            return x_out, convert(eltype(z), Inf)
        else
            return x_out, convert(eltype(z), -Inf)
        end
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

    S_chol = length(_S) == 1 ? _S[1] : cholesky!(_S)
    rdiv!(K, S_chol)

    loglikelihood = zero(eltype(K))
    loglikelihood = pn_logpdf!(measurement, S_chol, C_d)

    # x_out.μ .= m_p .+ K * (0 .- z)
    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)

    # M_cache .= I(D) .- mul!(M_cache, K, H)
    _matmul!(M_cache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        M_cache[i, i] += 1
    end

    fast_X_A_Xt!(x_out.Σ, P_p, M_cache)

    if !isnothing(R)
        # M = Matrix(x_out.Σ) + K * Matrix(R) * K'
        _matmul!(M_cache, x_out.Σ.R', x_out.Σ.R)
        _matmul!(K1_cache, K, R.R')
        _matmul!(M_cache, K1_cache, K1_cache', 1, 1)
        chol = cholesky!(Symmetric(M_cache), check=false)
        if issuccess(chol)
            copy!(x_out.Σ.R, chol.U)
        else
            x_out.Σ.R .= triangularize!([x_out.Σ.R; K1_cache']; cachemat=M_cache)
        end
    end

    return x_out, loglikelihood
end
function pn_logpdf!(measurement, S_chol, tmpmean)
    μ = reshape(measurement.μ, :)
    Σ = S_chol

    d = length(μ)
    z = ldiv!(Σ, copy!(tmpmean, μ))

    return -0.5 * μ'z - 0.5 * d * log(2π) - 0.5 * logdet(Σ)
end

# Kronecker version
function update!(
    x_out::SRGaussian{T,<:IsometricKroneckerProduct},
    x_pred::SRGaussian{T,<:IsometricKroneckerProduct},
    measurement::Gaussian{
        T,
        <:AbstractVector,
        <:Union{<:PSDMatrix{T,<:IsometricKroneckerProduct},<:IsometricKroneckerProduct},
    },
    H::IsometricKroneckerProduct,
    K1_cache::IsometricKroneckerProduct,
    K2_cache::IsometricKroneckerProduct,
    M_cache::IsometricKroneckerProduct,
    C_dxd::IsometricKroneckerProduct,
    C_d::AbstractVector;
    R::Union{Nothing,PSDMatrix{T,<:IsometricKroneckerProduct}}=nothing,
) where {T}
    D = length(x_out.μ)  # full_state_dim
    d = H.rdim           # ode_dimension_dim
    Q = D ÷ d            # n_derivatives_dim
    _x_out = Gaussian{T}(reshape_no_alloc(x_out.μ, d, Q)', PSDMatrix(x_out.Σ.R.B))
    _x_pred = Gaussian{T}(reshape_no_alloc(x_pred.μ, d, Q)', PSDMatrix(x_pred.Σ.R.B))
    _measurement = Gaussian{T}(
        reshape_no_alloc(measurement.μ, d, 1)',
        measurement.Σ isa PSDMatrix ? PSDMatrix(measurement.Σ.R.B) : measurement.Σ.B,
    )
    _H = H.B
    _K1_cache = K1_cache.B
    _K2_cache = K2_cache.B
    _M_cache = M_cache.B
    _C_dxd = C_dxd.B
    _R = isnothing(R) ? nothing : PSDMatrix(R.R.B)

    _, loglikelihood = update!(
        _x_out,
        _x_pred,
        _measurement,
        _H,
        _K1_cache,
        _K2_cache,
        _M_cache,
        _C_dxd,
        C_d,
        R=_R,
    )
    return x_out, loglikelihood
end

function update!(
    x_out::SRGaussian{T,<:BlocksOfDiagonals},
    x_pred::SRGaussian{T,<:BlocksOfDiagonals},
    measurement::Gaussian{
        T,
        <:AbstractVector,
        <:Union{<:PSDMatrix{T,<:BlocksOfDiagonals},<:BlocksOfDiagonals},
    },
    H::BlocksOfDiagonals,
    K1_cache::BlocksOfDiagonals,
    K2_cache::BlocksOfDiagonals,
    M_cache::BlocksOfDiagonals,
    C_dxd::BlocksOfDiagonals,
    C_d::AbstractVector;
    R::Union{Nothing,PSDMatrix{T,<:BlocksOfDiagonals}}=nothing,
) where {T}
    d = length(blocks(x_out.Σ.R))
    q = size(blocks(x_out.Σ.R)[1], 1) - 1

    ll = zero(eltype(x_out.μ))
    @views for i in eachindex(blocks(x_out.Σ.R))
        _, _ll = update!(
            Gaussian{T}(x_out.μ[i:d:end],
                PSDMatrix(x_out.Σ.R.blocks[i])),
            Gaussian{T}(x_pred.μ[i:d:end],
                PSDMatrix(x_pred.Σ.R.blocks[i])),
            Gaussian{T}(measurement.μ[i:d:end],
                if measurement.Σ isa PSDMatrix
                    PSDMatrix(measurement.Σ.R.blocks[i])
                else
                    measurement.Σ.blocks[i]
                end),
            H.blocks[i],
            K1_cache.blocks[i],
            K2_cache.blocks[i],
            M_cache.blocks[i],
            C_dxd.blocks[i],
            view(C_d, i:i);
            R=isnothing(R) ? nothing : PSDMatrix(blocks(R.R)[i]),
        )
        ll += _ll
    end
    return x_out, ll
end

# Short-hand with cache
function update!(x_out, x, measurement, H; cache, R=nothing)
    @unpack K1, m_tmp, C_DxD, C_dxd, C_Dxd, C_d = cache
    K2 = C_Dxd
    return update!(x_out, x, measurement, H, K1, K2, C_DxD, C_dxd, C_d; R)
end
