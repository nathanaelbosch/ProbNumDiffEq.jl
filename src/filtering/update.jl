"""
    update(x, measurement, H; R=nothing)

Update step in Kalman filtering for linear dynamics models.

Given a Gaussian ``x = \\mathcal{N}(μ, Σ)``
and a measurement ``z = \\mathcal{N}(\\hat{z}, S)``, with ``S = H Σ H^T + R``,
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
When called with a `PSDMatrix` covariance it performs the update in Joseph / square-root
form.

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
function update(x::SRGaussian, measurement::Gaussian, H::AbstractMatrix; R=nothing)
    m, C = mean(x), cov(x)
    z, S = mean(measurement), cov(measurement)

    K = Matrix(C) * H' * inv(S)
    m_new = m - K * z
    C_new = X_A_Xt(C, (I - K * H))
    isnothing(R) || (C_new = PSDMatrix(qr([C_new.R; R.R * K']).R))

    return Gaussian(m_new, C_new)
end

"""
    update(x, obs::LinearizedObservation)

Condition `x` on the observation `obs`, with dense copies of all matrices. This is
`update(x, measurement, H)` with the measurement ``\\mathcal{N}(\\hat{z}, S)``, where
``\\hat{z} = r + H (μ - m)`` and ``S = H Σ H^T + R``.
"""
function update(x::Gaussian, obs::LinearizedObservation)
    x = Gaussian(mean(x), Matrix(cov(x)))
    return update(x, _dense_measurement(x, obs)...)
end
function update(x::SRGaussian, obs::LinearizedObservation)
    x = Gaussian(mean(x), PSDMatrix(Matrix(x.Σ.R)))
    return update(x, _dense_measurement(x, obs)...; R=obs.R)
end
function _dense_measurement(x, obs)
    μ, Σ, H = mean(x), Matrix(cov(x)), Matrix(obs.H)
    S = H * Σ * H'
    isnothing(obs.R) || (S += Matrix(obs.R))
    return Gaussian(obs.r + H * (μ - obs.m), S), H
end

"""
    update!(x_out, x_pred, obs::LinearizedObservation; cache)

Condition the Gaussian `x_pred` on the observation `obs` and write the result into `x_out`,
with the square-root Kalman update in Joseph form:
```math
\\begin{aligned}
\\hat{z} &= r + H (μ - m), \\\\
S &= H Σ H^T + R, \\\\
K &= Σ H^T S^{-1}, \\\\
μ^F &= μ - K \\hat{z}, \\\\
Σ^F &= (I - K H) Σ (I - K H)^T + K R K^T.
\\end{aligned}
```
Return `(; loglikelihood, ztSinvz, S)`: the log-likelihood of the observation,
``\\hat{z}^T S^{-1} \\hat{z}``, and ``S``.

This is [`update_mean!`](@ref) followed by [`update_cov!`](@ref).

See also: [`update`](@ref).
"""
function update!(x_out, x_pred, obs::LinearizedObservation; cache)
    (; loglikelihood, ztSinvz, S, K, B) = update_mean!(x_out, x_pred, obs; cache)
    update_cov!(x_out, x_pred, obs, K, B; cache)
    return (; loglikelihood, ztSinvz, S)
end

"""
    update!(x, obs::LinearizedObservation; cache)

Condition the Gaussian `x` on the observation `obs` in place, with the buffers of the
solver's `cache`. `x` is first copied to `cache.x_tmp`, so neither `x` nor `obs.m` may be
stored there, but `obs.m` may be `x.μ`. An observation of size 0 leaves `x` unchanged.
"""
function update!(x, obs::LinearizedObservation; cache)
    o = length(obs.r)
    if o == 0
        T = eltype(x.μ)
        return (; loglikelihood=zero(T), ztSinvz=zero(T), S=nothing)
    end
    x_pred = copy!(cache.x_tmp, x)
    if obs.m === x.μ
        obs = LinearizedObservation(x_pred.μ, obs.r, obs.H, obs.R)
    end
    return update!(x, x_pred, obs; cache=make_obssized_cache(cache; o))
end

"""
    make_obssized_cache(cache; o)

The buffers of `cache` that [`update!`](@ref) uses, sized for an `o`-dimensional
observation: `cache` itself if `o` is the dimension `d` of the ODE, otherwise
views (dense covariance) or the first `o` blocks (block-diagonal covariance). The Kronecker
covariance only supports `o = d`.
"""
function make_obssized_cache(cache; o)
    if o == cache.d
        return cache
    else
        return make_obssized_cache(cache.covariance_factorization, cache; o)
    end
end
function make_obssized_cache(::DenseCovariance, cache; o)
    @unpack K1, C_DxD, C_dxd, C_Dxd, C_d, measurement = cache
    return (
        K1=view(K1, :, 1:o),
        C_dxd=view(C_dxd, 1:o, 1:o),
        C_Dxd=view(C_Dxd, :, 1:o),
        C_d=view(C_d, 1:o),
        C_DxD=C_DxD,
        measurement=Gaussian(view(measurement.μ, 1:o), view(measurement.Σ, 1:o, 1:o)),
    )
end
function make_obssized_cache(::BlockDiagonalCovariance, cache; o)
    @unpack K1, C_DxD, C_dxd, C_Dxd, C_d, measurement = cache
    first_blocks(M) = BlocksOfDiagonals(blocks(M)[1:o])
    return (
        K1=first_blocks(K1),
        C_dxd=first_blocks(C_dxd),
        C_Dxd=first_blocks(C_Dxd),
        C_d=view(C_d, 1:o),
        C_DxD=first_blocks(C_DxD),
        measurement=Gaussian(view(measurement.μ, 1:o), first_blocks(measurement.Σ)),
    )
end

"""
    update_mean!(x_out, x_pred, obs::LinearizedObservation; cache)
    update_mean!(x_out, x_pred, obs::LinearizedObservation,
                 K1_cache, K2_cache, measurement_cache, C_dxd, C_d)

The mean part of [`update!`](@ref): write ``μ^F`` into `x_out.μ` and return
`(; loglikelihood, ztSinvz, S, K, B)`, with the gain `K` and `B = x_pred.Σ.R * H'` for
[`update_cov!`](@ref). An iterated update calls it for every linearization, and
[`update_cov!`](@ref) only for the last one. `obs.r` must not be stored in `x_out.μ` or
`cache.C_d`, which serve as workspace.
"""
function update_mean!(
    x_out,
    x_pred,
    obs::LinearizedObservation,
    K1_cache,
    K2_cache,
    measurement_cache,
    C_dxd,
    C_d,
)
    (; m, H, R) = obs
    # r + H (μ - m) = r - H (m - μ)
    z = if m === x_pred.μ
        obs.r
    else
        Δ = x_out.μ .= m .- x_pred.μ
        measurement_cache.μ .= obs.r .- _matmul!(C_d, H, Δ)
    end
    return _update_mean!(x_out, x_pred, z, H, R, K1_cache, K2_cache, measurement_cache.Σ,
        C_dxd, C_d)
end
function update_mean!(x_out, x_pred, obs::LinearizedObservation; cache)
    @unpack K1, C_Dxd, measurement, C_dxd, C_d = cache
    return update_mean!(x_out, x_pred, obs, K1, C_Dxd, measurement, C_dxd, C_d)
end

function _update_mean!(
    x_out::SRGaussian,
    x_pred::SRGaussian,
    z::AbstractVecOrMat,
    H::AbstractMatrix,
    R::Union{Nothing,PSDMatrix},
    K1_cache::AbstractMatrix,
    K2_cache::AbstractMatrix,
    S_cache::AbstractMatrix,
    C_dxd::AbstractMatrix,
    C_d::AbstractArray,
)
    m_p, P_p = x_pred.μ, x_pred.Σ

    # S = K1ᵀ K1 + R with K1 = √Σ Hᵀ
    K1 = _matmul!(K1_cache, P_p.R, H')
    S = _matmul!(S_cache, K1', K1)
    isnothing(R) || add!(S, _matmul!(C_dxd, R.R', R.R))

    if (isnothing(R) || iszero(R)) && iszero(P_p)
        copy!(x_out.μ, m_p)
        loglikelihood = convert(eltype(z), iszero(z) ? Inf : -Inf)
        ztSinvz = convert(eltype(z), iszero(z) ? 0 : Inf)
        return (; loglikelihood, ztSinvz, S, K=fill!(K2_cache, 0), B=K1)
    end

    _S = make_hermitian_if_fowarddiff(copy!(C_dxd, S))
    S_chol = length(_S) == 1 ? _S[1] : cholesky!(_S)
    K = rdiv!(_matmul!(K2_cache, P_p.R', K1), S_chol)

    (; loglikelihood, ztSinvz) = pn_logpdf!(C_d, z, S_chol)

    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)
    return (; loglikelihood, ztSinvz, S, K, B=K1)
end
function pn_logpdf!(v, z, S_chol)
    μ = reshape(z, :)
    w = ldiv!(S_chol, copy!(reshape(v, :), μ))
    n = length(μ)
    ztSinvz = μ'w
    # logdet(S ⊗ I_k) = k logdet(S), with k = n ÷ size(S, 1)
    loglikelihood =
        -0.5 * ztSinvz - 0.5 * n * log(2π) -
        0.5 * (n ÷ size(S_chol, 1)) * logdet(S_chol)
    return (; loglikelihood, ztSinvz)
end
function _update_mean!(
    x_out::SRGaussian{T,<:IsometricKroneckerProduct},
    x_pred::SRGaussian{T,<:IsometricKroneckerProduct},
    z::AbstractVector,
    H::IsometricKroneckerProduct,
    R::Union{Nothing,KroneckerPSD},
    K1_cache::IsometricKroneckerProduct,
    K2_cache::IsometricKroneckerProduct,
    S_cache::IsometricKroneckerProduct,
    C_dxd::IsometricKroneckerProduct,
    C_d::AbstractVector,
) where {T}
    d = H.rdim
    args = (x_out, x_pred, z, H, R, K1_cache, K2_cache, S_cache, C_dxd)
    # `C_d` is not converted: it holds the flattened residual
    (; loglikelihood, ztSinvz) =
        _update_mean!(map(x -> _kronecker_factor(x, d), args)..., C_d)
    return (; loglikelihood, ztSinvz, S=S_cache, K=K2_cache, B=K1_cache)
end
function _update_mean!(
    x_out::SRGaussian{T,<:BlocksOfDiagonals},
    x_pred::SRGaussian{T,<:BlocksOfDiagonals},
    z::AbstractVector,
    H::BlocksOfDiagonals,
    R::Union{Nothing,BlocksOfDiagonalsPSD},
    K1_cache::BlocksOfDiagonals,
    K2_cache::BlocksOfDiagonals,
    S_cache::BlocksOfDiagonals,
    C_dxd::BlocksOfDiagonals,
    C_d::AbstractVector,
) where {T}
    d = nblocks(H)
    args = (x_out, x_pred, z, H, R, K1_cache, K2_cache, S_cache, C_dxd, C_d)
    loglikelihood, ztSinvz = zero(T), zero(T)
    for i in 1:d
        block = _update_mean!(map(x -> _diagonal_block(x, i, d), args)...)
        loglikelihood += block.loglikelihood
        ztSinvz += block.ztSinvz
    end
    return (; loglikelihood, ztSinvz, S=S_cache, K=K2_cache, B=K1_cache)
end
function _update_mean!(
    x_out::SRGaussian{T,<:BlocksOfDiagonals},
    x_pred::SRGaussian{T,<:BlocksOfDiagonals},
    z::AbstractVector,
    H::BlocksOfScaledSelections,
    R::Union{Nothing,BlocksOfDiagonalsPSD},
    K1_cache::BlocksOfDiagonals,
    K2_cache::BlocksOfDiagonals,
    S_cache::BlocksOfDiagonals,
    C_dxd::BlocksOfDiagonals,
    C_d::AbstractVector,
) where {T}
    d, o = nblocks(x_out.Σ.R), length(H.selected_dimensions)
    o < d && copy!(x_out.μ, x_pred.μ)
    args = (z, H, R, K1_cache, K2_cache, S_cache, C_dxd, C_d)
    loglikelihood, ztSinvz = zero(T), zero(T)
    for (k, i) in enumerate(H.selected_dimensions)
        block = _update_mean!(
            _diagonal_block(x_out, i, d), _diagonal_block(x_pred, i, d),
            map(x -> _diagonal_block(x, k, o), args)...)
        loglikelihood += block.loglikelihood
        ztSinvz += block.ztSinvz
    end
    return (; loglikelihood, ztSinvz, S=S_cache, K=K2_cache, B=K1_cache)
end

"""
    update_cov!(x_out, x_pred, obs::LinearizedObservation, K, B; cache)
    update_cov!(x_out, x_pred, obs::LinearizedObservation, K, B, M_cache, KR_cache)

The covariance part of [`update!`](@ref): write ``Σ^F`` into `x_out.Σ`, with the `K` and `B`
that [`update_mean!`](@ref) returned for the same `x_pred` and `obs`.
"""
function update_cov!(x_out, x_pred, obs::LinearizedObservation, K, B, M_cache, KR_cache)
    _update_cov!(x_out.Σ, x_pred.Σ, obs.H, obs.R, K, B, M_cache, KR_cache)
    return x_out
end
update_cov!(x_out, x_pred, obs::LinearizedObservation, K, B; cache) =
    update_cov!(x_out, x_pred, obs, K, B, cache.C_DxD, cache.K1)

function _update_cov!(
    Σ_out::PSDMatrix,
    Σ_pred::PSDMatrix,
    H::AbstractMatrix,
    R::Union{Nothing,PSDMatrix},
    K::AbstractMatrix,
    B::AbstractMatrix,
    M_cache::AbstractMatrix,
    KR_cache::AbstractMatrix,
)
    if (isnothing(R) || iszero(R)) && iszero(Σ_pred)
        copy!(Σ_out.R, Σ_pred.R)
        return Σ_out
    end

    # √Σ (I - K H)ᵀ = √Σ - B Kᵀ
    _matmul!(copy!(Σ_out.R, Σ_pred.R), B, K', -1.0, 1.0)

    if !isnothing(R)
        # Σ^F = (I - K H) Σ (I - K H)ᵀ + KR KRᵀ with KR = K √Rᵀ
        M = _matmul!(M_cache, Σ_out.R', Σ_out.R)
        KR = _matmul!(KR_cache, K, R.R')
        _matmul!(M, KR, KR', 1, 1)
        chol = cholesky!(Symmetric(M), check=false)
        if issuccess(chol)
            copy!(Σ_out.R, chol.U)
        else
            Σ_out.R .= triangularize!([Σ_out.R; KR']; cachemat=M)
        end
    end

    return Σ_out
end
function _update_cov!(
    Σ_out::KroneckerPSD,
    Σ_pred::KroneckerPSD,
    H::IsometricKroneckerProduct,
    R::Union{Nothing,KroneckerPSD},
    K::IsometricKroneckerProduct,
    B::IsometricKroneckerProduct,
    M_cache::IsometricKroneckerProduct,
    KR_cache::IsometricKroneckerProduct,
)
    on_kronecker_factors(
        _update_cov!, H.rdim, Σ_out, Σ_pred, H, R, K, B, M_cache, KR_cache)
    return Σ_out
end
function _update_cov!(
    Σ_out::BlocksOfDiagonalsPSD,
    Σ_pred::BlocksOfDiagonalsPSD,
    H::BlocksOfDiagonals,
    R::Union{Nothing,BlocksOfDiagonalsPSD},
    K::BlocksOfDiagonals,
    B::BlocksOfDiagonals,
    M_cache::BlocksOfDiagonals,
    KR_cache::BlocksOfDiagonals,
)
    foreach_diagonal_block(
        _update_cov!, nblocks(H), Σ_out, Σ_pred, H, R, K, B, M_cache, KR_cache)
    return Σ_out
end
function _update_cov!(
    Σ_out::BlocksOfDiagonalsPSD,
    Σ_pred::BlocksOfDiagonalsPSD,
    H::BlocksOfScaledSelections,
    R::Union{Nothing,BlocksOfDiagonalsPSD},
    K::BlocksOfDiagonals,
    B::BlocksOfDiagonals,
    M_cache::BlocksOfDiagonals,
    KR_cache::BlocksOfDiagonals,
)
    d, o = nblocks(Σ_out.R), length(H.selected_dimensions)
    o < d && copy!(Σ_out.R, Σ_pred.R)
    args = (H, R, K, B, M_cache, KR_cache)
    for (k, i) in enumerate(H.selected_dimensions)
        _update_cov!(_diagonal_block(Σ_out, i, d), _diagonal_block(Σ_pred, i, d),
            map(x -> _diagonal_block(x, k, o), args)...)
    end
    return Σ_out
end
