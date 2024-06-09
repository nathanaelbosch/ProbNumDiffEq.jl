############################################################################################
# `Gaussian` distributions
# Based on @mschauer's GaussianDistributions.jl
############################################################################################
"""
    Gaussian(μ, Σ) -> P

Gaussian distribution with mean `μ` and covariance `Σ`. Defines `rand(P)` and `(log-)pdf(P, x)`.
Designed to work with `Number`s, `UniformScaling`s, `StaticArrays` and `PSD`-matrices.

Implementation details: On `Σ` the functions `logdet`, `whiten` and `unwhiten`
(or `chol` as fallback for the latter two) are called.
"""
struct Gaussian{T,S}
    μ::T
    Σ::S
    Gaussian{T,S}(μ, Σ) where {T,S} = new(μ, Σ)
    Gaussian(μ::T, Σ::S) where {T,S} = new{T,S}(μ, Σ)
end
Base.convert(::Type{Gaussian{T,S}}, g::Gaussian) where {T,S} =
    Gaussian(convert(T, g.μ), convert(S, g.Σ))

# Base
Base.:(==)(g1::Gaussian, g2::Gaussian) = g1.μ == g2.μ && g1.Σ == g2.Σ
Base.isapprox(g1::Gaussian, g2::Gaussian; kwargs...) =
    isapprox(g1.μ, g2.μ; kwargs...) && isapprox(g1.Σ, g2.Σ; kwargs...)
copy(P::Gaussian) = Gaussian(copy(P.μ), copy(P.Σ))
similar(P::Gaussian) = Gaussian(similar(P.μ), similar(P.Σ))
Base.copyto!(P::AbstractArray{<:Gaussian}, idx::Integer, el::Gaussian) = begin
    P[idx] = copy(el)
    P
end
function Base.copy!(dst::Gaussian, src::Gaussian)
    copy!(dst.μ, src.μ)
    copy!(dst.Σ, src.Σ)
    return dst
end
Base.iterate(::Gaussian) = error()
Base.iterate(::Gaussian, s) = error()
Base.length(P::Gaussian) = length(mean(P))
size(g::Gaussian) = size(g.μ)
Base.eltype(::Type{G}) where {G<:Gaussian} = G
Base.@propagate_inbounds Base.getindex(P::Gaussian, i::Integer) =
    Gaussian(P.μ[i], diag(P.Σ)[i])

# Statistics
mean(P::Gaussian) = P.μ
cov(P::Gaussian) = P.Σ
var(P::Gaussian{<:Number}) = P.Σ
std(P::Gaussian{<:Number}) = sqrt(var(P))
var(g::Gaussian) = diag(g.Σ)
std(g::Gaussian) = sqrt.(diag(g.Σ))

dim(P::Gaussian) = length(P.μ)
ndims(g::Gaussian) = ndims(g.μ)

# whiten(Σ::PSD, z) = Σ.σ\z
whiten(Σ, z) = cholesky(Σ).U' \ z
whiten(Σ::Number, z) = sqrt(Σ) \ z
whiten(Σ::UniformScaling, z) = sqrt(Σ.λ) \ z

# unwhiten(Σ::PSD, z) = Σ.σ*z
unwhiten(Σ, z) = cholesky(Σ).U' * z
unwhiten(Σ::Number, z) = sqrt(Σ) * z
unwhiten(Σ::UniformScaling, z) = sqrt(Σ.λ) * z

sqmahal(P::Gaussian, x) = norm_sqr(whiten(P.Σ, x - P.μ))

rand(P::Gaussian) = rand(GLOBAL_RNG, P)
rand(RNG::AbstractRNG, P::Gaussian) = P.μ + unwhiten(P.Σ, randn(RNG, typeof(P.μ)))
rand(RNG::AbstractRNG, P::Gaussian{Vector{T}}) where {T} =
    P.μ + unwhiten(P.Σ, randn(RNG, T, length(P.μ)))
rand(RNG::AbstractRNG, P::Gaussian{<:Number}) =
    P.μ + sqrt(P.Σ) * randn(RNG, typeof(one(P.μ)))

_logdet(Σ, d) = logdet(Σ)
_logdet(J::UniformScaling, d) = log(J.λ) * d
logpdf(P::Gaussian, x) = -(sqmahal(P, x) + _logdet(P.Σ, dim(P)) + dim(P) * log(2pi)) / 2
pdf(P::Gaussian, x) = exp(logpdf(P::Gaussian, x))
cdf(P::Gaussian{Number}, x) = Distributions.normcdf(P.μ, sqrt(P.Σ), x)

Base.:+(g::Gaussian, vec) = Gaussian(g.μ + vec, g.Σ)
Base.:+(vec, g::Gaussian) = g + vec
Base.:-(g::Gaussian, vec) = g + (-vec)
Base.:*(M, g::Gaussian) = Gaussian(M * g.μ, X_A_Xt(g.Σ, M))

function rand_scalar(RNG::AbstractRNG, P::Gaussian{T}, dims) where {T}
    X = zeros(T, dims)
    for i in 1:length(X)
        X[i] = rand(RNG, P)
    end
    X
end

function rand_vector(
    RNG::AbstractRNG,
    P::Gaussian{Vector{T}},
    dims::Union{Integer,NTuple},
) where {T}
    X = zeros(T, dim(P), dims...)
    for i in 1:prod(dims)
        X[:, i] = rand(RNG, P)
    end
    X
end
rand(RNG::AbstractRNG, P::Gaussian, dim::Integer) = rand_scalar(RNG, P, dim)
rand(RNG::AbstractRNG, P::Gaussian, dims::Tuple{Vararg{Int64,N}} where {N}) =
    rand_scalar(RNG, P, dims)

rand(RNG::AbstractRNG, P::Gaussian{Vector{T}}, dim::Integer) where {T} =
    rand_vector(RNG, P, dim)
rand(
    RNG::AbstractRNG,
    P::Gaussian{Vector{T}},
    dims::Tuple{Vararg{Int64,N}} where {N},
) where {T} = rand_vector(RNG, P, dims)
rand(P::Gaussian, dims::Tuple{Vararg{Int64,N}} where {N}) = rand(GLOBAL_RNG, P, dims)
rand(P::Gaussian, dim::Integer) = rand(GLOBAL_RNG, P, dim)

# RecursiveArrayTools
RecursiveArrayTools.recursivecopy(P::Gaussian) = copy(P)
RecursiveArrayTools.recursivecopy!(dst::Gaussian, src::Gaussian) = copy!(dst, src)

# Print
show(io::IO, g::Gaussian) = print(io, "Gaussian($(g.μ), $(g.Σ))")
show(io::IO, ::MIME"text/plain", g::Gaussian{T,S}) where {T,S} =
    print(io, "Gaussian{$T,$S}($(g.μ), $(g.Σ))")

############################################################################################
# `SRGaussian`: Gaussians with PDFMatrix covariances
############################################################################################
const SRGaussian{T,S} = Gaussian{VM,PSDMatrix{T,S}} where {VM<:AbstractVecOrMat{T}}
function _gaussian_mul!(g_out::SRGaussian, M::AbstractMatrix, g_in::SRGaussian)
    _matmul!(g_out.μ, M, g_in.μ)
    fast_X_A_Xt!(g_out.Σ, g_in.Σ, M)
    return g_out
end

const SRGaussianList{T,S} = StructArray{SRGaussian{T,S}}
mean(s::SRGaussianList) = s.μ
