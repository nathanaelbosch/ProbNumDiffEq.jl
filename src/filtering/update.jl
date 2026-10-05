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
`obs` is linearized at ``μ``. The square root of ``(I - K H) Σ (I - K H)^T`` is computed as
``\\sqrt{Σ} - B K^T`` with ``B = \\sqrt{Σ} H^T``, in ``O(D^2 o)`` for an `o`-dimensional
observation.

See also: [`update`](@ref).
"""
function update!(x_out, x_pred, obs::LinearizedObservation; cache)
    (; loglikelihood, S, K, B) = update_mean!(x_out, x_pred, obs; cache)
    update_cov!(x_out, x_pred, obs, K, B; cache)
    return (; loglikelihood, S)
end

"""
    update_mean!(x_out, x_pred, obs::LinearizedObservation; cache)
    update_mean!(x_out, x_pred, obs::LinearizedObservation,
                 K1_cache, K2_cache, measurement_cache, C_dxd, C_d)

The mean of [`update!`](@ref): write ``μ^F`` into `x_out.μ`, and return the named tuple
`(; loglikelihood, S, K, B)` with the gain `K` and `B = √Σ Hᵀ` for [`update_cov!`](@ref).
An iterated update calls it for each linearization and [`update_cov!`](@ref) only for the
last one. The second form takes the buffers explicitly; the first takes them from `cache`,
and writes `S` into `cache.measurement.Σ`, `K` into `cache.C_Dxd` and `B` into
`cache.K1`.
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
        return (; loglikelihood, S, K=fill!(K2_cache, 0), B=K1)
    end

    _S = make_hermitian_if_fowarddiff(copy!(C_dxd, S))
    S_chol = length(_S) == 1 ? _S[1] : cholesky!(_S)
    K = rdiv!(_matmul!(K2_cache, P_p.R', K1), S_chol)

    loglikelihood = pn_logpdf!(C_d, z, S_chol)

    x_out.μ .= m_p .- _matmul!(x_out.μ, K, z)
    return (; loglikelihood, S, K, B=K1)
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
    return (; loglikelihood, S=S_cache, K=K2_cache, B=K1_cache)
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
    return (; loglikelihood, S=S_cache, K=K2_cache, B=K1_cache)
end

"""
    update_cov!(x_out, x_pred, obs::LinearizedObservation, K, B; cache)
    update_cov!(x_out, x_pred, obs::LinearizedObservation, K, B, M_cache, KR_cache)

The covariance of [`update!`](@ref): write ``Σ^F`` into `x_out.Σ`, for the gain `K` and
`B = √Σ Hᵀ` that [`update_mean!`](@ref) returned for the same `x_pred` and `obs`.
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
        # Σ^F = √Σ^Fᵀ √Σ^F + KR KRᵀ with KR = K √Rᵀ
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

"""
    update(x, obs::LinearizedObservation)

Condition the Gaussian `x` on the observation `obs` and return the result, with dense
matrices: the allocating reference for [`update!`](@ref),
```math
\\begin{aligned}
S &= H Σ H^T + R, \\\\
K &= Σ H^T S^{-1}, \\\\
μ^F &= μ - K (z + H (μ - m)), \\\\
Σ^F &= Σ - K S K^T.
\\end{aligned}
```
"""
function update(x::Gaussian, obs::LinearizedObservation)
    μ, Σ = mean(x), Matrix(cov(x))
    H = Matrix(obs.H)
    S = H * Σ * H'
    isnothing(obs.R) || (S += Matrix(obs.R))
    K = Σ * H' / S
    return Gaussian(μ - K * (obs.z + H * (μ - obs.m)), Σ - K * S * K')
end
