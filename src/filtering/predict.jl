"""
    predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix)

Prediction step in Kalman filtering for linear dynamics models.

Given a Gaussian ``x = \\mathcal{N}(μ, Σ)``, compute and return
``\\mathcal{N}(A μ, A Σ A^T + Q)``.

See also the non-allocating square-root version [`predict!`](@ref).
"""
predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) =
    Gaussian(predict_mean(x, A), predict_cov(x, A, Q))
predict_mean(x::Gaussian, A::AbstractMatrix) = A * x.μ
predict_cov(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) = A * x.Σ * A' + Q
predict_cov(x::SRGaussian, A::AbstractMatrix, Q::PSDMatrix) = PSDMatrix(qr([x.Σ.R * A'; Q.R]).R)
predict_cov(x::SRGaussian{T,<:IKP}, A::IKP, Q::PSDMatrix{T,<:IKP}) where {T} =
    IsoKroneckerProduct(x.Σ.R.alpha, x.Σ.R.ldim, predict_cov(Gaussian(x.μ, PSDMatrix(x.Σ.R.B)), A.B, PSDMatrix(Q.R.B)))

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
    predict_mean!(x_out, x_curr, Ah)
    predict_cov!(x_out, x_curr, Ah, Qh, C_DxD, C_2DxD, diffusion)
    return x_out
end

function predict_mean!(x_out::Gaussian, x_curr::Gaussian, Ah::AbstractMatrix)
    D = length(x_out.μ)
    a = size(Ah, 1)
    d = D ÷ a
    _matmul!(reshape_no_alloc(x_out.μ, a, d), Ah, reshape_no_alloc(x_curr.μ, a, d))
    return x_out.μ
end

function predict_cov!(
    x_out::SRGaussian,
    x_curr::SRGaussian,
    Ah::AbstractMatrix,
    Qh::PSDMatrix,
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    diffusion=1,
)
    if iszero(diffusion)
        fast_X_A_Xt!(x_out.Σ, x_curr.Σ, Ah)
        return x_out.Σ
    end
    R, M = C_2DxD, C_DxD
    D, D = size(Qh)

    _matmul!(view(R, 1:D, 1:D), x_curr.Σ.R, Ah')
    _matmul!(view(R, D+1:2D, 1:D), Qh.R, sqrt.(diffusion))
    _matmul!(M, R', R)
    chol = cholesky!(Symmetric(M), check=false)

    Q_R = if issuccess(chol)
        alloc_free_get_U!(chol)
    else
        triangularize!(R, cachemat=C_DxD)
    end
    copy!(x_out.Σ.R, Q_R)
    return x_out.Σ
end

# Kronecker version
function predict_cov!(
    x_out::SRGaussian{T,<:IKP},
    x_curr::SRGaussian{T,<:IKP},
    Ah::IKP,
    Qh::PSDMatrix{S,<:IKP},
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    diffusion=1,
) where {T,S}
    _x_out = Gaussian(x_out.μ, PSDMatrix(x_out.Σ.R.B))
    _x_curr = Gaussian(x_curr.μ, PSDMatrix(x_curr.Σ.R.B))
    _Ah = Ah.B
    _Qh = PSDMatrix(Qh.R.B)
    _D = size(_Qh, 1)
    _C_DxD = view(C_DxD, 1:_D, 1:_D)
    _C_2DxD = view(C_2DxD, 1:2*_D, 1:_D)
    _diffusion = diffusion isa IKP ? diffusion.B : diffusion

    predict_cov!(
        _x_out,
        _x_curr,
        _Ah,
        _Qh,
        _C_DxD,
        _C_2DxD,
        _diffusion,
    )
end
