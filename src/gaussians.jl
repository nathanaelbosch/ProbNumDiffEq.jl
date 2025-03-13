############################################################################################
# `Gaussian` distributions
# Based on @mschauer's GaussianDistributions.jl
############################################################################################
import MarkovKernels: Normal
const Gaussian = Normal

function Normal(μ::AbstractVector, Σ::PSDMatrix)
    T = promote_type(eltype(μ), eltype(Σ))
    return Normal{T}(convert(AbstractVector{T}, μ), convert(PSDMatrix{T}, Σ))
end

############################################################################################
# `SRGaussian`: Gaussians with PDFMatrix covariances
############################################################################################
const SRGaussian{T,S} = Gaussian{A,VM,PSDMatrix{T,S}} where {A,VM<:AbstractVecOrMat{T}}
function _gaussian_mul!(g_out::SRGaussian, M::AbstractMatrix, g_in::SRGaussian)
    _matmul!(g_out.μ, M, g_in.μ)
    fast_X_A_Xt!(g_out.Σ, g_in.Σ, M)
    return g_out
end

const SRGaussianList{T,S} = StructArray{SRGaussian{T,S}}
mean(s::SRGaussianList) = s.μ
