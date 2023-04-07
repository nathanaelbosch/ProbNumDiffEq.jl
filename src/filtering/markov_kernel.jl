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

"""
    marginalize!(
        xout::Gaussian{Vector{T},PSDMatrix{T,S}}
        x::Gaussian{Vector{T},PSDMatrix{T,S}},
        K::AffineNormalKernel{<:AbstractMatrix,Union{<:Number,<:AbstractVector,Missing},<:PSDMatrix};
        C_DxD, C_3DxD[, diffusion=1]
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
function marginalize!(xout, x, K; C_DxD, C_3DxD, diffusion=1)
    marginalize_mean!(xout, x, K)
    marginalize_cov!(xout, x, K; C_DxD, C_3DxD, diffusion)
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
    C_3DxD::AbstractMatrix,
    diffusion=1,
)
    D = length(x_curr.μ)
    A, b, C = K
    R, M = C_3DxD, C_DxD

    if iszero(diffusion)
        fast_X_A_Xt!(x_out.Σ, x_curr.Σ, A)
        return x_out.Σ
    end

    _matmul!(view(R, 1:D, 1:D), x_curr.Σ.R, A')
    if !isone(diffusion)
        _matmul!(view(R, D+1:3D, 1:D), C.R, sqrt.(diffusion))
    else
        @.. R[D+1:3D, 1:D] = C.R
    end

    Q_R = begin
        _matmul!(M, R', R)
        chol = cholesky!(Symmetric(M), check=false)
        if issuccess(chol)
            chol.U
        else
            triangularize!(R, cachemat=C_DxD)
        end
    end
    copy!(x_out.Σ.R, Q_R)
    return x_out.Σ
end

"""
    compute_backward_kernel!(Kout, xpred, x, K; C_DxD, C_2DxD, cachemat[, diffusion=1])

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
x \\mid y \\sim \\mathcal{N} \\left( x; G x + d, Λ \\right),
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
cache objects `C_DxD`, `C_2DxD`, `cachemat`), so the actual formulas implemented here
differ a bit.

The resulting backward kernels are used to smooth the posterior, via [`marginalize!`](@ref).
"""
function compute_backward_kernel!(
    Kout::KT1,
    xpred::XT,
    x::XT,
    K::KT2;
    C_DxD::AbstractMatrix,
    C_2DxD::AbstractMatrix,
    cachemat::AbstractMatrix,
    diffusion=1,
) where {
    XT<:SRGaussian,
    KT1<:AffineNormalKernel{<:AbstractMatrix,<:AbstractVector,<:PSDMatrix},
    KT2<:AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix},
}
    # @assert Matrix(UpperTriangular(xpred.Σ.R)) == Matrix(xpred.Σ.R)

    D = length(x.μ)
    A, _, Q = K
    G, b, Λ = Kout

    # G = Matrix(x.Σ) * A' / Matrix(xpred.Σ)
    _matmul!(C_DxD, x.Σ.R, A')
    _matmul!(G, x.Σ.R', C_DxD)
    rdiv!(G, Cholesky(xpred.Σ.R, 'U', 0))

    # b = μ - G * μ_pred
    b .= x.μ .- _matmul!(b, G, xpred.μ)

    # Λ.R[1:D, 1:D] = (I - G * A)'
    _matmul!(view(Λ.R, 1:D, 1:D), A', G', -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        Λ.R[i, i] += 1
    end
    # Λ.R[D+1:2D, 1:D] = (G * Q.R')'
    if !isone(diffusion)
        _matmul!(C_DxD, Q.R, sqrt.(diffusion))
        _matmul!(view(Λ.R, D+1:2D, 1:D), C_DxD, G')
    else
        _matmul!(view(Λ.R, D+1:2D, 1:D), Q.R, G')
    end

    return Kout
end
