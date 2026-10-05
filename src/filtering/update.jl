"""
    LinearizedObservation(m, z, H, R=nothing)

An observation ``y = h(x) + v``, ``v \\sim \\mathcal{N}(0, R)``, of the filter state ``x``,
with ``h`` linearized at the point `m`:
```math
h(x) - y ≈ z + H (x - m),
```
where `z = h(m) - y` and `H` is the Jacobian of ``h`` at `m`. `R` is a `PSDMatrix`, or
`nothing` for an exact observation.

The ODE step observes ``y = 0`` through ``h(x) = E_1 x - f(E_0 x, t)`` and linearizes at the
predicted mean ``μ``, so `m = μ`, `z = h(μ)` and `H = E_1 - J E_0`. Data ``y`` observed
through a matrix ``C`` gives the linear ``h(x) = C E_0 x``, so `H = C E_0` and, with
`m = μ`, `z = C E_0 μ - y`. An iterated update linearizes at its current iterate instead.
A statistical linearization takes `m` as the mean of the distribution it linearizes over,
gives `z = E[h(x)] - y`, and adds the covariance of its linearization error to `R`.

[`update!`](@ref) conditions a state on it. It dispatches on the types of the state and of
`H`: an explicit matrix in the covariance structure of the state, or another type that
implements it, such as an operator that applies `H` and `H'` matrix-free.
"""
struct LinearizedObservation{mT,zT,HT,RT}
    m::mT
    z::zT
    H::HT
    R::RT
end
LinearizedObservation(m, z, H) = LinearizedObservation(m, z, H, nothing)

"""
    update!(x_out, x_pred, obs::LinearizedObservation; cache)

Condition the Gaussian `x_pred` on the observation `obs`, write the result into `x_out`, and
return the named tuple `(; loglikelihood, S)`: the log-likelihood of the observation and its
covariance `S = H Σ Hᵀ + R`, which is written into `cache.measurement.Σ`.

This is [`update_mean!`](@ref) followed by [`update_cov!`](@ref). For an explicit `H`, it is
the square-root Kalman update in Joseph form:
```math
\\begin{aligned}
S &= H Σ H^T + R, \\\\
K &= Σ H^T S^{-1}, \\\\
μ^F &= μ - K (z + H (μ - m)), \\\\
Σ^F &= (I - K H) Σ (I - K H)^T + K R K^T.
\\end{aligned}
```
`z + H (μ - m)` is the value of the linearization at the predicted mean ``μ``; it is `z` when
`obs` is linearized at ``μ``.

See also: [`update`](@ref).
"""
function update!(x_out, x_pred, obs::LinearizedObservation; cache)
    (; loglikelihood, S, K) = update_mean!(x_out, x_pred, obs; cache)
    update_cov!(x_out, x_pred, obs, K; cache)
    return (; loglikelihood, S)
end

"""
    update_mean!(x_out, x_pred, obs::LinearizedObservation; cache)
    update_mean!(x_out, x_pred, obs::LinearizedObservation,
                 K1_cache, K2_cache, measurement_cache, C_dxd, C_d)

The mean of [`update!`](@ref): write ``μ^F`` into `x_out.μ`, and return the named tuple
`(; loglikelihood, S, K)` with the gain `K` for [`update_cov!`](@ref). An iterated update
calls it for each linearization and [`update_cov!`](@ref) only for the last one. The second
form takes the buffers explicitly; the first takes them from `cache`, and writes `S` into
`cache.measurement.Σ` and `K` into `cache.C_Dxd`.
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
    # z + H (μ - m) = z - H (m - μ), with x_out.μ and C_d as buffers
    z = if m === x_pred.μ
        obs.z
    else
        Δ = x_out.μ .= m .- x_pred.μ
        measurement_cache.μ .= obs.z .- _matmul!(C_d, H, Δ)
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
        return (; loglikelihood, S, K=fill!(K2_cache, 0))
    end

    _S = make_hermitian_if_fowarddiff(copy!(C_dxd, S))
    S_chol = length(_S) == 1 ? _S[1] : cholesky!(_S)
    K = rdiv!(_matmul!(K2_cache, P_p.R', K1), S_chol)

    loglikelihood = pn_logpdf!(C_d, z, S_chol)

    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)
    return (; loglikelihood, S, K)
end
function pn_logpdf!(v, z, S_chol)
    μ = reshape(z, :)
    w = ldiv!(S_chol, copy!(reshape(v, :), μ))
    n = length(μ)
    # With a Kronecker covariance `S ⊗ I`, `μ` stacks `n ÷ size(S, 1)` independent columns
    return -0.5 * μ'w - 0.5 * n * log(2π) - 0.5 * (n ÷ size(S_chol, 1)) * logdet(S_chol)
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
    # `C_d` is workspace for the flattened residual, so it is not converted
    (; loglikelihood) =
        _update_mean!(map(x -> _kronecker_factor(x, d), args)..., C_d)
    return (; loglikelihood, S=S_cache, K=K2_cache)
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
    loglikelihood = zero(T)
    for i in 1:d
        loglikelihood +=
            _update_mean!(map(x -> _diagonal_block(x, i, d), args)...).loglikelihood
    end
    return (; loglikelihood, S=S_cache, K=K2_cache)
end

"""
    update_cov!(x_out, x_pred, obs::LinearizedObservation, K; cache)
    update_cov!(x_out, x_pred, obs::LinearizedObservation, K, M_cache, K1_cache)

The covariance of [`update!`](@ref): write ``Σ^F`` into `x_out.Σ`, for the gain `K` that
[`update_mean!`](@ref) returned for the same `x_pred` and `obs`.
"""
function update_cov!(x_out, x_pred, obs::LinearizedObservation, K, M_cache, K1_cache)
    _update_cov!(x_out.Σ, x_pred.Σ, obs.H, obs.R, K, M_cache, K1_cache)
    return x_out
end
update_cov!(x_out, x_pred, obs::LinearizedObservation, K; cache) =
    update_cov!(x_out, x_pred, obs, K, cache.C_DxD, cache.K1)

function _update_cov!(
    Σ_out::PSDMatrix,
    Σ_pred::PSDMatrix,
    H::AbstractMatrix,
    R::Union{Nothing,PSDMatrix},
    K::AbstractMatrix,
    M_cache::AbstractMatrix,
    K1_cache::AbstractMatrix,
)
    if (isnothing(R) || iszero(R)) && iszero(Σ_pred)
        copy!(Σ_out.R, Σ_pred.R)
        return Σ_out
    end

    # M = I - K H
    M = _matmul!(M_cache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in axes(M, 1)
        M[i, i] += 1
    end
    fast_X_A_Xt!(Σ_out, Σ_pred, M)

    if !isnothing(R)
        # Σ^F = √Σ^Fᵀ √Σ^F + B Bᵀ with B = K √Rᵀ
        _matmul!(M, Σ_out.R', Σ_out.R)
        B = _matmul!(K1_cache, K, R.R')
        _matmul!(M, B, B', 1, 1)
        chol = cholesky!(Symmetric(M), check=false)
        if issuccess(chol)
            copy!(Σ_out.R, chol.U)
        else
            Σ_out.R .= triangularize!([Σ_out.R; B']; cachemat=M)
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
    M_cache::IsometricKroneckerProduct,
    K1_cache::IsometricKroneckerProduct,
)
    on_kronecker_factors(_update_cov!, H.rdim, Σ_out, Σ_pred, H, R, K, M_cache, K1_cache)
    return Σ_out
end
function _update_cov!(
    Σ_out::BlocksOfDiagonalsPSD,
    Σ_pred::BlocksOfDiagonalsPSD,
    H::BlocksOfDiagonals,
    R::Union{Nothing,BlocksOfDiagonalsPSD},
    K::BlocksOfDiagonals,
    M_cache::BlocksOfDiagonals,
    K1_cache::BlocksOfDiagonals,
)
    foreach_diagonal_block(
        _update_cov!, nblocks(H), Σ_out, Σ_pred, H, R, K, M_cache, K1_cache)
    return Σ_out
end

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

    if (isnothing(R) || iszero(R)) && iszero(P_p)
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

    _S = make_hermitian_if_fowarddiff(_S)
    S_chol = length(_S) == 1 ? _S[1] : cholesky!(_S)
    rdiv!(K, S_chol)

    loglikelihood = zero(eltype(K))
    loglikelihood = pn_logpdf_old!(measurement, S_chol, C_d)

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
function pn_logpdf_old!(measurement, S_chol, tmpmean)
    μ = reshape(measurement.μ, :)
    Σ = S_chol

    d = length(μ)
    z = ldiv!(Σ, copy!(tmpmean, μ))

    # With a Kronecker covariance `S ⊗ I`, `μ` stacks `d ÷ size(S, 1)` independent columns
    return -0.5 * μ'z - 0.5 * d * log(2π) - 0.5 * (d ÷ size(Σ, 1)) * logdet(Σ)
end

function update!(
    x_out::SRGaussian{T,<:IsometricKroneckerProduct},
    x_pred::SRGaussian{T,<:IsometricKroneckerProduct},
    measurement::Gaussian{
        <:AbstractVector,<:Union{<:KroneckerPSD{T},<:IsometricKroneckerProduct}},
    H::IsometricKroneckerProduct,
    K1_cache::IsometricKroneckerProduct,
    K2_cache::IsometricKroneckerProduct,
    M_cache::IsometricKroneckerProduct,
    C_dxd::IsometricKroneckerProduct,
    C_d::AbstractVector;
    R::Union{Nothing,KroneckerPSD{T}}=nothing,
) where {T}
    d = H.rdim
    args = (x_out, x_pred, measurement, H, K1_cache, K2_cache, M_cache, C_dxd)
    # `C_d` is workspace for the flattened measurement, so it is not converted
    _, loglikelihood = update!(
        map(x -> _kronecker_factor(x, d), args)..., C_d; R=_kronecker_factor(R, d))
    return x_out, loglikelihood
end
function update!(
    x_out::SRGaussian{T,<:BlocksOfDiagonals},
    x_pred::SRGaussian{T,<:BlocksOfDiagonals},
    measurement::Gaussian{
        <:AbstractVector,<:Union{<:BlocksOfDiagonalsPSD{T},<:BlocksOfDiagonals}},
    H::BlocksOfDiagonals,
    K1_cache::BlocksOfDiagonals,
    K2_cache::BlocksOfDiagonals,
    M_cache::BlocksOfDiagonals,
    C_dxd::BlocksOfDiagonals,
    C_d::AbstractVector;
    R::Union{Nothing,BlocksOfDiagonalsPSD{T}}=nothing,
) where {T}
    d = nblocks(H)
    args = (x_out, x_pred, measurement, H, K1_cache, K2_cache, M_cache, C_dxd, C_d)
    loglikelihood = zero(T)
    for i in 1:d
        _, ll = update!(
            map(x -> _diagonal_block(x, i, d), args)...; R=_diagonal_block(R, i, d))
        loglikelihood += ll
    end
    return x_out, loglikelihood
end

# Short-hand with cache
function update!(x_out, x, measurement, H; cache, R=nothing)
    @unpack K1, m_tmp, C_DxD, C_dxd, C_Dxd, C_d = cache
    K2 = C_Dxd
    return update!(x_out, x, measurement, H, K1, K2, C_DxD, C_dxd, C_d; R)
end
