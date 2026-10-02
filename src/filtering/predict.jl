"""
    predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix)

Prediction step in Kalman filtering for linear dynamics models.

Given a Gaussian ``x = \\mathcal{N}(μ, Σ)``, compute and return
``\\mathcal{N}(A μ, A Σ A^T + Q)``.

See also the non-allocating square-root version [`predict!`](@ref).
"""
predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) =
    Gaussian(predict_mean(x.μ, A), predict_cov(x.Σ, A, Q))
predict_mean(μ::AbstractVecOrMat, A::AbstractMatrix) = A * μ
predict_cov(Σ::AbstractMatrix, A::AbstractMatrix, Q::AbstractMatrix) = A * Σ * A' + Q
predict_cov(Σ::PSDMatrix, A::AbstractMatrix, Q::PSDMatrix) =
    PSDMatrix(qr([Σ.R * A'; Q.R]).R)
predict_cov(
    Σ::PSDMatrix{T,<:IsometricKroneckerProduct},
    A::IsometricKroneckerProduct,
    Q::PSDMatrix{T,<:IsometricKroneckerProduct},
) where {T} = begin
    P_pred_breve = predict_cov(PSDMatrix(Σ.R.B), A.B, PSDMatrix(Q.R.B))
    return PSDMatrix(IsometricKroneckerProduct(Σ.R.rdim, P_pred_breve.R))
end
predict_cov(
    Σ::PSDMatrix{T,<:BlocksOfDiagonals},
    A::BlocksOfDiagonals,
    Q::PSDMatrix{S,<:BlocksOfDiagonals},
) where {T,S} = begin
    R_blocks = map(blocks(Σ.R), blocks(A), blocks(Q.R)) do Σ_R_i, A_i, Q_R_i
        predict_cov(PSDMatrix(Σ_R_i), A_i, PSDMatrix(Q_R_i)).R
    end
    return PSDMatrix(BlocksOfDiagonals(R_blocks))
end

"""
    predict!(x_out, x_curr, Ah, Qh, cachemat)

In-place and square-root implementation of [`predict`](@ref)
which saves the result into `x_out`.

Only works with `PSDMatrices.PSDMatrix` types as `Ah`, `Qh`, and in the
covariances of `x_curr` and `x_out` (both of type `Gaussian`).
To prevent allocations, a cache matrix `cachemat` of size ``D \\times 2D``
(where ``D \\times D`` is the size of `Ah` and `Qh`) needs to be passed.

See also: [`predict`](@ref).
"""
function predict!(
    x_out::SRGaussian,
    x_curr::SRGaussian,
    Ah::AbstractMatrix,
    Qh::PSDMatrix,
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    diffusion=1,
)
    predict_mean!(x_out.μ, x_curr.μ, Ah)
    predict_cov!(x_out.Σ, x_curr.Σ, Ah, Qh, C_DxD, C_2DxD, diffusion)
    return x_out
end

function predict_mean!(
    m_out::AbstractVecOrMat,
    m_curr::AbstractVecOrMat,
    Ah::AbstractMatrix,
)
    _matmul!(m_out, Ah, m_curr)
    return m_out
end

function predict_cov!(
    Σ_out::PSDMatrix,
    Σ_curr::PSDMatrix,
    Ah::AbstractMatrix,
    Qh::PSDMatrix,
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    diffusion::Union{Number,Diagonal},
)
    if iszero(diffusion)
        fast_X_A_Xt!(Σ_out, Σ_curr, Ah)
        return Σ_out
    end
    R, M = C_2DxD, C_DxD
    D = size(Qh, 1)

    _matmul!(view(R, 1:D, 1:D), Σ_curr.R, Ah')
    if isone(diffusion)
        @.. R[(D+1):2D, 1:D] = Qh.R
    else
        apply_diffusion!(PSDMatrix(view(R, (D+1):2D, 1:D)), Qh, diffusion)
    end
    _matmul!(M, R', R)
    chol = cholesky!(Symmetric(M), check=false)

    Q_R = if issuccess(chol) && is_well_conditioned(chol.U)
        chol.U
    else
        triangularize!(R, cachemat=C_DxD)
    end
    copy!(Σ_out.R, Q_R)
    return Σ_out
end

function predict_cov!(
    Σ_out::KroneckerPSD{T},
    Σ_curr::KroneckerPSD{T},
    Ah::IsometricKroneckerProduct,
    Qh::KroneckerPSD{S},
    C_DxD::IsometricKroneckerProduct,
    C_2DxD::IsometricKroneckerProduct,
    diffusion::Union{Number,Diagonal},
) where {T,S}
    on_kronecker_factors(
        predict_cov!, Ah.rdim, Σ_out, Σ_curr, Ah, Qh, C_DxD, C_2DxD, diffusion)
    return Σ_out
end
function predict_cov!(
    Σ_out::BlocksOfDiagonalsPSD{T},
    Σ_curr::BlocksOfDiagonalsPSD{T},
    Ah::BlocksOfDiagonals,
    Qh::BlocksOfDiagonalsPSD{S},
    C_DxD::BlocksOfDiagonals,
    C_2DxD::BlocksOfDiagonals,
    diffusion::Union{Number,Diagonal},
) where {T,S}
    foreach_diagonal_block(
        predict_cov!, nblocks(Ah), Σ_out, Σ_curr, Ah, Qh, C_DxD, C_2DxD, diffusion)
    return Σ_out
end
