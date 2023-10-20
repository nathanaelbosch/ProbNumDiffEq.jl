############################################################################################
# Useful things when working with GaussianDistributions.Gaussian
############################################################################################
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
similar(P::Gaussian) = Gaussian(similar(P.μ), similar(P.Σ))
function Base.copy!(dst::Gaussian, src::Gaussian)
    copy!(dst.μ, src.μ)
    copy!(dst.Σ, src.Σ)
    return dst
end
RecursiveArrayTools.recursivecopy(P::Gaussian) = copy(P)
RecursiveArrayTools.recursivecopy!(dst::Gaussian, src::Gaussian) = copy!(dst, src)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T,S}) where {T,S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)
ndims(g::Gaussian) = ndims(g.μ)
var(g::Gaussian) = diag(g.Σ)
std(g::Gaussian) = sqrt.(diag(g.Σ))

############################################################################################
# `SRGaussian`: Gaussians with PDFMatrix covariances
############################################################################################
const SRGaussian{T,S} = Gaussian{<:AbstractVecOrMat{T},PSDMatrix{T,S}}
Base.:*(M::AbstractMatrix, g::SRGaussian) = Gaussian(M * g.μ, X_A_Xt(g.Σ, M))
# GaussianDistributions.whiten(Σ::PSDMatrix, z) = Σ.L\z

function _gaussian_mul!(g_out::SRGaussian, M::AbstractMatrix, g_in::SRGaussian)
    _matmul!(g_out.μ, M, g_in.μ)
    fast_X_A_Xt!(g_out.Σ, g_in.Σ, M)
    return g_out
end

const SRGaussianList{T,S} = StructArray{SRGaussian{T,S}}
mean(s::SRGaussianList) = s.μ
