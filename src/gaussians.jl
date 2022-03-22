# const PSDGaussian{T} = Gaussian{Vector{T}, PSDMatrix{T}}
# const PSDGaussianList{T} = StructArray{PSDGaussian{T}}
const SRGaussian{T,S} = Gaussian{Vector{T},SRMatrix{T,S}}
const SRGaussianList{T,S} = StructArray{SRGaussian{T,S}}

copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
similar(P::Gaussian) = Gaussian(similar(P.μ), similar(P.Σ))
function Base.copy!(dst::Gaussian, src::Gaussian)
    copy!(dst.μ, src.μ)
    copy!(dst.Σ, src.Σ)
    return dst
end
RecursiveArrayTools.recursivecopy(P::Gaussian) = copy(P)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T,S}) where {T,S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)
ndims(g::Gaussian) = ndims(g.μ)

Base.:*(M, g::SRGaussian) = Gaussian(M * g.μ, X_A_Xt(g.Σ, M))
# GaussianDistributions.whiten(Σ::SRMatrix, z) = Σ.L\z

function _gaussian_mul!(g_out::SRGaussian, M::AbstractMatrix, g_in::SRGaussian)
    _matmul!(g_out.μ, M, g_in.μ)
    X_A_Xt!(g_out.Σ, g_in.Σ, M)
    return g_out
end

var(p::SRGaussian) = diag(p.Σ)
std(p::SRGaussian) = sqrt.(var(p))
mean(s::SRGaussianList) = s.μ
var(s::SRGaussianList) = diag.(s.Σ)
std(s::SRGaussianList) = map(std, s)
