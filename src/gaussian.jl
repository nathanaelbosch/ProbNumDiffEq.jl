"""
    Gaussian(μ::AbstractVector, Σ::AbstractMatrix)

Multivariate Gaussian distribution ``\\mathcal{N}(\\mu, \\Sigma)``.

**Note:* There is currently no additional functionality implemented.
In the future we might instead use Distributions.jl.
"""
mutable struct Gaussian{T<:AbstractFloat}
    μ::AbstractVector{T}
    Σ::AbstractMatrix{T}
    Gaussian{T}(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:AbstractFloat} =
        (length(μ) == size(Σ,1)) ?
        (Σ≈Σ' ? new(μ, Symmetric(Σ)) : error("Σ is not symmetric: $Σ")) :
        # (Σ≈Σ' ? new(μ, Σ) : error("Σ is not symmetric: $Σ")) :
        error("Wrong input dimensions: size(μ)=$(size(μ)), size(Σ)=$(size(Σ))")
end
Gaussian(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Gaussian{T}(μ, Σ)
copy(g::Gaussian) = Gaussian(g.μ, g.Σ)
