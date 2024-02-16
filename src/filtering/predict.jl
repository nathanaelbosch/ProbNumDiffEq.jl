"""
    predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix)

Prediction step in Kalman filtering for linear dynamics models.

Given a Gaussian ``x = \\mathcal{N}(μ, Σ)``, compute and return
``\\mathcal{N}(A μ, A Σ A^T + Q)``.

See also the non-allocating square-root version [`predict!`](@ref).
"""
predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) =
    Gaussian(predict_mean(x.μ, A), predict_cov(x.Σ, A, Q))
predict_mean(μ::AbstractVector, A::AbstractMatrix) = A * μ
predict_cov(Σ::AbstractMatrix, A::AbstractMatrix, Q::AbstractMatrix) = A * Σ * A' + Q
predict_cov(Σ::PSDMatrix, A::AbstractMatrix, Q::PSDMatrix) =
    PSDMatrix(qr([Σ.R * A'; Q.R]).R)
predict_cov(
    Σ::PSDMatrix{T,<:IsometricKroneckerProduct},
    A::IsometricKroneckerProduct,
    Q::PSDMatrix{T,<:IsometricKroneckerProduct},
) where {T} = begin
    P_pred_breve = predict_cov(PSDMatrix(Σ.R.B), A.B, PSDMatrix(Q.R.B))
    return PSDMatrix(IsometricKroneckerProduct(Σ.R.ldim, P_pred_breve.R))
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
        @.. R[D+1:2D, 1:D] = Qh.R
    else
        apply_diffusion!(PSDMatrix(view(R, D+1:2D, 1:D)), Qh, diffusion)
    end
    _matmul!(M, R', R)
    chol = cholesky!(Symmetric(M), check=false)

    Q_R = if issuccess(chol)
        chol.U
    else
        triangularize!(R, cachemat=C_DxD)
    end
    copy!(Σ_out.R, Q_R)
    return Σ_out
end

# Kronecker version
function predict_cov!(
    Σ_out::PSDMatrix{T,<:IsometricKroneckerProduct},
    Σ_curr::PSDMatrix{T,<:IsometricKroneckerProduct},
    Ah::IsometricKroneckerProduct,
    Qh::PSDMatrix{S,<:IsometricKroneckerProduct},
    C_DxD::IsometricKroneckerProduct,
    C_2DxD::IsometricKroneckerProduct,
    diffusion::Union{Number,Diagonal},
) where {T,S}
    _Σ_out = PSDMatrix(Σ_out.R.B)
    _Σ_curr = PSDMatrix(Σ_curr.R.B)
    _Ah = Ah.B
    _Qh = PSDMatrix(Qh.R.B)
    _C_DxD = C_DxD.B
    _C_2DxD = C_2DxD.B
    _diffusion =
        diffusion isa Number ? diffusion :
        diffusion isa IsometricKroneckerProduct ? diffusion.B : diffusion

    return predict_cov!(_Σ_out, _Σ_curr, _Ah, _Qh, _C_DxD, _C_2DxD, _diffusion)
end

# BlockDiagonal version
function predict_cov!(
    Σ_out::PSDMatrix{T,<:BlockDiag},
    Σ_curr::PSDMatrix{T,<:BlockDiag},
    Ah::BlockDiag,
    Qh::PSDMatrix{S,<:BlockDiag},
    C_DxD::BlockDiag,
    C_2DxD::BlockDiag,
    diffusion::Union{Number,Diagonal},
) where {T,S}
    @simd ivdep for i in eachindex(blocks(Σ_out.R))
        predict_cov!(
            PSDMatrix(Σ_out.R.blocks[i]),
            PSDMatrix(Σ_curr.R.blocks[i]),
            Ah.blocks[i],
            PSDMatrix(Qh.R.blocks[i]),
            C_DxD.blocks[i],
            C_2DxD.blocks[i],
            diffusion isa Number ? diffusion : diffusion.diag[i],
        )
    end
    return Σ_out
end
