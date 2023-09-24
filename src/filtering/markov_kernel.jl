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

# Little bit of type piracy here:
isapprox(M1::PSDMatrix, M2::PSDMatrix; kwargs...) = isapprox(M1.R, M2.R; kwargs...)

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
    marginalize_mean!(xout, x, K)
    marginalize_cov!(xout, x, K; C_DxD, C_3DxD)
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
)
    _D = size(x_curr.Σ, 1)
    A, b, C = K
    R, M = C_3DxD, C_DxD

    _matmul!(view(R, 1:_D, 1:_D), x_curr.Σ.R, A')
    @.. R[_D+1:3_D, 1:_D] = C.R

    Q_R = triangularize!(R, cachemat=C_DxD)
    copy!(x_out.Σ.R, Q_R)
    return x_out.Σ
end

function marginalize_cov!(
    x_out::SRGaussian{T,<:Kronecker.KroneckerProduct},
    x_curr::SRGaussian{T,<:Kronecker.KroneckerProduct},
    K::AffineNormalKernel{
        <:AbstractMatrix,
        <:Any,
        <:PSDMatrix{S,<:Kronecker.KroneckerProduct},
    };
    C_DxD::AbstractMatrix,
    C_3DxD::AbstractMatrix,
) where {T,S}
    _x_out = Gaussian(x_out.μ, PSDMatrix(x_out.Σ.R.B))
    _x_curr = Gaussian(x_curr.μ, PSDMatrix(x_curr.Σ.R.B))
    _K = AffineNormalKernel(K.A.B, K.b, PSDMatrix(K.C.R.B))
    _D = size(_x_out.Σ, 1)
    _C_DxD = view(C_DxD, 1:_D, 1:_D)
    _C_3DxD = view(C_3DxD, 1:3*_D, 1:_D)
    marginalize_cov!(_x_out, _x_curr, _K; C_DxD=_C_DxD, C_3DxD=_C_3DxD)
end

"""
    compute_backward_kernel!(Kout, xpred, x, K; C_DxD[, diffusion=1])

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
cache `C_DxD`), so the actual formulas implemented here differ a bit.

The resulting backward kernels are used to smooth the posterior, via [`marginalize!`](@ref).
"""
function compute_backward_kernel!(
    Kout::KT1,
    xpred::XT,
    x::XT,
    K::KT2;
    C_DxD::AbstractMatrix,
    diffusion=1,
) where {
    XT<:SRGaussian,
    KT1<:AffineNormalKernel{<:AbstractMatrix,<:AbstractVector,<:PSDMatrix},
    KT2<:AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix},
}
    # @assert Matrix(UpperTriangular(xpred.Σ.R)) == Matrix(xpred.Σ.R)

    A, _, Q = K
    G, b, Λ = Kout

    D = length(x.μ)
    _D = size(G, 1)
    _a = D ÷ _D

    # G = Matrix(x.Σ) * A' / Matrix(xpred.Σ)
    _matmul!(C_DxD, x.Σ.R, A')
    _matmul!(G, x.Σ.R', C_DxD)
    rdiv!(G, Cholesky(xpred.Σ.R, 'U', 0))

    # b = μ - G * μ_pred
    _matmul!(reshape_no_alloc(b, _D, _a), G, reshape_no_alloc(xpred.μ, _D, _a))
    b .= x.μ .- b

    # Λ.R[1:D, 1:D] = x.Σ.R * (I - G * A)'
    _matmul!(C_DxD, A', G', -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:_D
        C_DxD[i, i] += 1
    end
    _matmul!(view(Λ.R, 1:_D, 1:_D), x.Σ.R, C_DxD)
    # Λ.R[D+1:2D, 1:D] = (G * Q.R')'
    if !isone(diffusion)
        _matmul!(C_DxD, Q.R, sqrt.(diffusion))
        _matmul!(view(Λ.R, _D+1:2_D, 1:_D), C_DxD, G')
    else
        _matmul!(view(Λ.R, _D+1:2_D, 1:_D), Q.R, G')
    end

    return Kout
end

function compute_backward_kernel!(
    Kout::KT1,
    xpred::XT,
    x::XT,
    K::KT2;
    C_DxD::AbstractMatrix,
    diffusion=1,
) where {
    XT<:SRGaussian{<:Number,<:Kronecker.KroneckerProduct},
    KT1<:AffineNormalKernel{<:AbstractMatrix,<:AbstractVector,<:PSDMatrix},
    KT2<:AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix},
}
    _Kout = AffineNormalKernel(Kout.A.B, Kout.b, PSDMatrix(Kout.C.R.B))
    _x_pred = Gaussian(xpred.μ, PSDMatrix(xpred.Σ.R.B))
    _x = Gaussian(x.μ, PSDMatrix(x.Σ.R.B))
    _K = AffineNormalKernel(K.A.B, K.b, PSDMatrix(K.C.R.B))
    _D = size(_Kout.A, 1)
    _C_DxD = view(C_DxD, 1:_D, 1:_D)
    compute_backward_kernel!(_Kout, _x_pred, _x, _K; C_DxD=_C_DxD, diffusion=diffusion)
end
