# const PSDGaussian{T} = Gaussian{Vector{T}, PSDMatrix{T}}
# const PSDGaussianList{T} = StructArray{PSDGaussian{T}}
const SRGaussian{T, S, M} = Gaussian{Vector{T}, SRMatrix{T, S, M}}
const SRGaussianList{T, S, M} = StructArray{SRGaussian{T, S, M}}

copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
copy!(dst::Gaussian, src::Gaussian) = (copy!(dst.μ, src.μ); copy!(dst.Σ, src.Σ); dst)
RecursiveArrayTools.recursivecopy(P::Gaussian) = copy(P)
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T, S}) where {T, S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")
size(g::Gaussian) = size(g.μ)
ndims(g::Gaussian) = ndims(g.μ)


Base.:*(M, g::SRGaussian) = Gaussian(M * g.μ, X_A_Xt(g.Σ, M))
# GaussianDistributions.whiten(Σ::SRMatrix, z) = Σ.L\z

function mul!(g_out::SRGaussian, M::AbstractMatrix, g_in::SRGaussian)
    mul!(g_out.μ, M, g_in.μ)
    X_A_Xt!(g_out.Σ, g_in.Σ, M)
    return g_out
end


var(p::SRGaussian{T}) where {T} = diag(p.Σ)
std(p::SRGaussian{T}) where {T} = sqrt.(diag(p.Σ))
mean(s::SRGaussianList{T}) where {T} = s.μ
var(s::SRGaussianList{T}) where {T} = diag.(s.Σ)
std(s::SRGaussianList{T}) where {T} = map(v -> sqrt.(v), (diag.(s.Σ)))
