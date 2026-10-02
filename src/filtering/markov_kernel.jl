"""
    AffineNormalKernel(A[, b], C)

Structure to represent affine Normal Markov kernels, i.e. conditional distributions of the
form
```math
\\begin{aligned}
y \\mid x \\sim \\mathcal{N} \\left( y; A x + b, C \\right).
\\end{aligned}
```

At the point of writing, `AffineNormalKernel`s are only used to precompute and store the
backward representation of the posterior (via [`compute_backward_kernel!`](@ref)) and for
smoothing (via [`marginalize!`](@ref)).
"""
struct AffineNormalKernel{TA,Tb,TC}
    A::TA
    b::Tb
    C::TC
end
AffineNormalKernel(A, C) = AffineNormalKernel(A, missing, C)

iterate(K::AffineNormalKernel, args...) = iterate((K.A, K.b, K.C), args...)

similar(K::AffineNormalKernel) =
    AffineNormalKernel(similar(K.A), ismissing(K.b) ? missing : similar(K.b), similar(K.C))
copy(K::AffineNormalKernel) =
    AffineNormalKernel(copy(K.A), ismissing(K.b) ? missing : copy(K.b), copy(K.C))
copy!(dst::AffineNormalKernel, src::AffineNormalKernel) = begin
    copy!(dst.A, src.A)
    copy!(dst.b, src.b)
    copy!(dst.C, src.C)
    return nothing
end

RecursiveArrayTools.recursivecopy(K::AffineNormalKernel) = copy(K)
RecursiveArrayTools.recursivecopy!(
    dst::AffineNormalKernel, src::AffineNormalKernel) = copy!(dst, src)

isapprox(K1::AffineNormalKernel, K2::AffineNormalKernel; kwargs...) =
    isapprox(K1.A, K2.A; kwargs...) &&
    isapprox(K1.b, K2.b; kwargs...) &&
    isapprox(K1.C, K2.C; kwargs...)
==(K1::AffineNormalKernel, K2::AffineNormalKernel) =
    K1.A == K2.A && K1.b == K2.b && K1.C == K2.C

"""
    marginalize!(
        xout::Gaussian{Vector{T},PSDMatrix{T,S}}
        x::Gaussian{Vector{T},PSDMatrix{T,S}},
        K::AffineNormalKernel{<:AbstractMatrix,Union{<:Number,<:AbstractVector,Missing},<:PSDMatrix};
        C_DxD, C_3DxD
    )

Basically the same as [`predict!`](@ref)), but in kernel language and with support for
affine transitions. At the time of writing, this is only used to smooth the posterior
using it's backward representation, where the kernels are precomputed with
[`compute_backward_kernel!`](@ref).

Note that this function assumes certain shapes:
- `size(x.μ) == (D, D)`
- `size(x.Σ) == (D, D)`
- `size(K.A) == (D, D)`
- `size(K.b) == (D,)`, or `missing`
- `size(K.C) == (D, D)`, _but with a tall square-root `size(K.C.R) == (3D, D)`
`xout` is assumes to have the same shapes as `x`.
"""
function marginalize!(xout, x, K; C_DxD, C_3DxD)
    marginalize_mean!(xout.μ, x.μ, K)
    marginalize_cov!(xout.Σ, x.Σ, K; C_DxD, C_3DxD)
end
function marginalize_mean!(
    μout::AbstractVecOrMat,
    μ::AbstractVecOrMat,
    K::AffineNormalKernel,
)
    _matmul!(μout, K.A, μ)
    if !ismissing(K.b)
        μout .+= K.b
    end
    return μout
end

function marginalize_cov!(
    Σ_out::PSDMatrix,
    Σ_curr::PSDMatrix,
    K::AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix};
    C_DxD::AbstractMatrix,
    C_3DxD::AbstractMatrix,
)
    _D = size(Σ_curr, 1)
    A, _, C = K
    R = C_3DxD

    _matmul!(view(R, 1:_D, 1:_D), Σ_curr.R, A')
    @.. R[(_D+1):3_D, 1:_D] = C.R

    Q_R = triangularize!(R, cachemat=C_DxD)
    copy!(Σ_out.R, Q_R)
    return Σ_out
end

function marginalize_cov!(
    Σ_out::KroneckerPSD{T},
    Σ_curr::KroneckerPSD{T},
    K::AffineNormalKernel{<:AbstractMatrix,<:Any,<:KroneckerPSD{S}};
    C_DxD::AbstractMatrix,
    C_3DxD::AbstractMatrix,
) where {T,S}
    # The covariance does not depend on `K.b`, and reshaping it would allocate
    _K = AffineNormalKernel(K.A, K.C)
    on_kronecker_factors(
        marginalize_cov!, Σ_out.R.rdim, Σ_out, Σ_curr, _K; C_DxD, C_3DxD)
    return Σ_out
end
function marginalize_cov!(
    Σ_out::BlocksOfDiagonalsPSD{T},
    Σ_curr::BlocksOfDiagonalsPSD{T},
    K::AffineNormalKernel{<:AbstractMatrix,<:Any,<:BlocksOfDiagonalsPSD{S}};
    C_DxD::AbstractMatrix,
    C_3DxD::AbstractMatrix,
) where {T,S}
    foreach_diagonal_block(
        marginalize_cov!, nblocks(Σ_out.R), Σ_out, Σ_curr, K; C_DxD, C_3DxD)
    return Σ_out
end

"""
    compute_backward_kernel!(Kout, xpred, x, K; C_DxD, C_2DxD, C_2Dx2D[, diffusion=1])

Compute the backward representation of the posterior, i.e. the conditional
distribution of the current state given the next state and the transition kernel.

More precisely, given a distribution (`x`)
```math
\\begin{aligned}
x \\sim \\mathcal{N} \\left( x; μ, Σ \\right),
\\end{aligned}
```
a kernel (`K`)
```math
\\begin{aligned}
y \\mid x \\sim \\mathcal{N} \\left( y; A x + b, C \\right),
\\end{aligned}
```
and a distribution (`xpred`) obtained via marginalization
```math
\\begin{aligned}
y &\\sim \\mathcal{N} \\left( y; μ^P, Σ^P \\right), \\\\
μ^P &= A μ + b, \\\\
Σ^P &= A Σ A^\\top + C,
\\end{aligned}
```
this function computes the conditional distribution
```math
\\begin{aligned}
x \\mid y \\sim \\mathcal{N} \\left( x; G y + d, Λ \\right),
\\end{aligned}
```
where
```math
\\begin{aligned}
G &= Σ A^\\top (Σ^P)^{-1}, \\\\
d &= μ - G μ^P, \\\\
Λ &= Σ - G Σ^P G^\\top.
\\end{aligned}
```
Everything is computed in square-root form and with minimal allocations (thus the
cache `C_DxD`), so the actual formulas implemented here differ a bit. These formulas divide by
`xpred.Σ`, which squares its condition number; if `xpred.Σ` is ill-conditioned (see
`is_well_conditioned`), the kernel is instead computed from a QR decomposition, in the
caches `C_2DxD` and `C_2Dx2D`.

The resulting backward kernels are used to smooth the posterior, via [`marginalize!`](@ref).
"""
function compute_backward_kernel!(
    Kout::KT1,
    xpred::XT,
    x::XT,
    K::KT2;
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    C_2Dx2D::AbstractMatrix,
    diffusion=1,
) where {
    XT<:SRGaussian,
    KT1<:AffineNormalKernel{<:AbstractMatrix,<:AbstractVecOrMat,<:PSDMatrix},
    KT2<:AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix},
}
    # @assert Matrix(UpperTriangular(xpred.Σ.R)) == Matrix(xpred.Σ.R)

    if !is_well_conditioned(xpred.Σ.R)
        return qr_backward_kernel!(Kout, xpred, x, K; C_2DxD, C_2Dx2D, diffusion)
    end

    A, _, Q = K
    G, b, Λ = Kout

    D = output_dim = size(G, 1)

    # G = Matrix(x.Σ) * A' / Matrix(xpred.Σ)
    _matmul!(C_DxD, x.Σ.R, A')
    _matmul!(G, x.Σ.R', C_DxD)
    if !iszero(G) # TODO check if this is actually correct
        rdiv!(G, Cholesky(xpred.Σ.R, 'U', 0))
    end

    # b = μ - G * μ_pred
    _matmul!(b, G, xpred.μ)
    b .= x.μ .- b

    # Λ.R[1:D, 1:D] = x.Σ.R * (I - G * A)'
    _matmul!(C_DxD, A', G', -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        C_DxD[i, i] += 1
    end
    _matmul!(view(Λ.R, 1:D, 1:D), x.Σ.R, C_DxD)
    # Λ.R[D+1:2D, 1:D] = (G * Q.R')'
    if isone(diffusion)
        _matmul!(view(Λ.R, (D+1):2D, 1:D), Q.R, G')
    else
        apply_diffusion!(PSDMatrix(C_DxD), Q, diffusion)
        _matmul!(view(Λ.R, (D+1):2D, 1:D), C_DxD, G')
    end

    return Kout
end

# For an ill-conditioned prediction, compute the kernel from the QR decomposition
# [R Aᵀ R; R_C 0] = U [R₁ R₁₂; 0 R₂] instead, where Σ = RᵀR and C = R_CᵀR_C: then
# G = (R₁⁻¹ R₁₂)ᵀ and Λ = R₂ᵀR₂, without forming a covariance
function qr_backward_kernel!(Kout, xpred, x, K; C_2DxD, C_2Dx2D, diffusion)
    A, _, C = K
    G, b, Λ = Kout
    D = size(G, 1)

    M = C_2Dx2D
    _matmul!(view(M, 1:D, 1:D), x.Σ.R, A')
    @.. M[1:D, (D+1):2D] = x.Σ.R
    apply_diffusion!(PSDMatrix(view(M, (D+1):2D, 1:D)), C, diffusion)
    fill!(view(M, (D+1):2D, (D+1):2D), zero(eltype(M)))
    # `C_2DxD` serves as the LAPACK workspace, which needs 2D columns
    triangularize!(M; cachemat=reshape(C_2DxD, D, 2D))

    R12 = view(M, 1:D, (D+1):2D)
    ldiv!(UpperTriangular(view(M, 1:D, 1:D)), R12)
    G .= transpose(R12)

    _matmul!(b, G, xpred.μ)
    b .= x.μ .- b

    copy!(view(Λ.R, 1:D, :), view(M, (D+1):2D, (D+1):2D))
    fill!(view(Λ.R, (D+1):2D, :), zero(eltype(Λ.R)))
    return Kout
end

function compute_backward_kernel!(
    Kout::AffineNormalKernel{
        <:IsometricKroneckerProduct,
        <:AbstractVector,
        <:KroneckerPSD{T},
    },
    xpred::SRGaussian{T,<:IsometricKroneckerProduct},
    x::SRGaussian{T,<:IsometricKroneckerProduct},
    K::AffineNormalKernel{<:IsometricKroneckerProduct,<:Any,<:KroneckerPSD{T}};
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    C_2Dx2D::AbstractMatrix,
    diffusion::Union{Number,Diagonal}=1,
) where {T}
    on_kronecker_factors(
        compute_backward_kernel!, K.A.rdim, Kout, xpred, x, K;
        C_DxD, C_2DxD, C_2Dx2D, diffusion)
    return Kout
end
function compute_backward_kernel!(
    Kout::AffineNormalKernel{
        <:BlocksOfDiagonals,
        <:AbstractVector,
        <:BlocksOfDiagonalsPSD{T},
    },
    xpred::SRGaussian{T,<:BlocksOfDiagonals},
    x::SRGaussian{T,<:BlocksOfDiagonals},
    K::AffineNormalKernel{<:BlocksOfDiagonals,<:Any,<:BlocksOfDiagonalsPSD{T}};
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    C_2Dx2D::AbstractMatrix,
    diffusion::Union{Number,Diagonal}=1,
) where {T}
    foreach_diagonal_block(
        compute_backward_kernel!, nblocks(K.A), Kout, xpred, x, K;
        C_DxD, C_2DxD, C_2Dx2D, diffusion)
    return Kout
end
