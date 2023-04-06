struct AffineNormalKernel{TA,Tb,TC}
    A::TA
    b::Tb
    C::TC
end
AffineNormalKernel(A, C) = AffineNormalKernel(A, missing, C)

copy(K::AffineNormalKernel) =
    if !ismissing(K.b)
        AffineNormalKernel(copy(K.A), copy(K.b), copy(K.C))
    else
        AffineNormalKernel(copy(K.A), missing, copy(K.C))
    end

RecursiveArrayTools.recursivecopy(K::AffineNormalKernel) = copy(K)
RecursiveArrayTools.recursivecopy!(
    dst::AffineNormalKernel, src::AffineNormalKernel) = copy!(dst, src)

isapprox(K1::AffineNormalKernel, K2::AffineNormalKernel; kwargs...) =
    isapprox(K1.A, K2.A; kwargs...) &&
    isapprox(K1.b, K2.b; kwargs...) &&
    isapprox(K1.C, K2.C; kwargs...)

function marginalize!(xout, x, K; C_DxD, C_2DxD, diffusion=1)
    marginalize_mean!(xout, x, K)
    marginalize_cov!(xout, x, K; C_DxD, C_2DxD, diffusion)
end
function marginalize_mean!(xout::Gaussian, x::Gaussian, K::AffineNormalKernel)
    _matmul!(xout.μ, K.A, x.μ)
    if !ismissing(K.b)
        xout.μ .+= K.b
    end
    return xout.μ
end

function marginalize_cov!(
    x_out::SRGaussian,
    x_curr::SRGaussian,
    K::AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix};
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    diffusion=1,
)
    if iszero(diffusion)
        fast_X_A_Xt!(x_out.Σ, x_curr.Σ, K.A)
        return x_out.Σ
    end
    R, M = C_2DxD, C_DxD
    D, D = size(K.C)

    _matmul!(view(R, 1:D, 1:D), x_curr.Σ.R, K.A')
    _matmul!(view(R, D+1:2D, 1:D), K.C.R, sqrt.(diffusion))
    _matmul!(M, R', R)
    chol = cholesky!(Symmetric(M), check=false)

    Q_R = if issuccess(chol)
        chol.U
    else
        triangularize!(R, cachemat=C_DxD)
    end
    copy!(x_out.Σ.R, Q_R)
    return x_out.Σ
end

function compute_backward_kernel!(
    Kout::KT1,
    xpred::XT,
    x::XT,
    K::KT2;
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    cachemat::AbstractMatrix,
) where {
    XT<:SRGaussian,
    KT1<:AffineNormalKernel{<:AbstractMatrix,<:AbstractVector,<:PSDMatrix},
    KT2<:AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix},
}
    D = length(x.μ)

    # G = Matrix(x.Σ) * K.A' / Matrix(xpred.Σ)
    _matmul!(C_DxD, x.Σ.R, K.A')
    _matmul!(Kout.A, x.Σ.R', C_DxD)
    # @assert Matrix(UpperTriangular(xpred.Σ.R)) == Matrix(xpred.Σ.R)
    rdiv!(Kout.A, Cholesky(xpred.Σ.R, 'U', 0))
    G = Kout.A

    # b =
    Kout.b .= x.μ .- _matmul!(Kout.b, G, xpred.μ)

    # M = [(I - G*K.A)'; (G*K.C.R')']
    _matmul!(C_DxD, K.A', G', -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        C_DxD[i, i] += 1
    end
    _matmul!(view(C_2DxD, 1:D, 1:D), x.Σ.R, C_DxD)
    _matmul!(view(C_2DxD, D+1:2D, 1:D), K.C.R, G')
    ΛR = triangularize!(C_2DxD; cachemat)
    copy!(Kout.C.R, ΛR)

    return Kout
end
