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
    @.. R[_D+1:3_D, 1:_D] = C.R

    Q_R = triangularize!(R, cachemat=C_DxD)
    copy!(Σ_out.R, Q_R)
    return Σ_out
end

function marginalize_cov!(
    Σ_out::PSDMatrix{T,<:IsometricKroneckerProduct},
    Σ_curr::PSDMatrix{T,<:IsometricKroneckerProduct},
    K::AffineNormalKernel{
        <:AbstractMatrix,
        <:Any,
        <:PSDMatrix{S,<:IsometricKroneckerProduct},
    };
    C_DxD::AbstractMatrix,
    C_3DxD::AbstractMatrix,
) where {T,S}
    @assert ismissing(K.b) || isnothing(K.b)
    _Σ_out = PSDMatrix(Σ_out.R.B)
    _Σ_curr = PSDMatrix(Σ_curr.R.B)
    _K = AffineNormalKernel(K.A.B, nothing, PSDMatrix(K.C.R.B))
    _D = size(_Σ_out, 1)
    _C_DxD = C_DxD.B
    _C_3DxD = C_3DxD.B
    return marginalize_cov!(_Σ_out, _Σ_curr, _K; C_DxD=_C_DxD, C_3DxD=_C_3DxD)
end

function marginalize_cov!(
    Σ_out::PSDMatrix{T,<:BlockDiag},
    Σ_curr::PSDMatrix{T,<:BlockDiag},
    K::AffineNormalKernel{
        <:AbstractMatrix,
        <:Any,
        <:PSDMatrix{S,<:BlockDiag},
    };
    C_DxD::AbstractMatrix,
    C_3DxD::AbstractMatrix,
) where {T,S}
    @assert ismissing(K.b) || isnothing(K.b)
    @inbounds @simd ivdep for i in eachindex(blocks(Σ_out.R))
        _Σ_out = PSDMatrix(Σ_out.R.blocks[i])
        _Σ_curr = PSDMatrix(Σ_curr.R.blocks[i])
        _K = AffineNormalKernel(K.A.blocks[i], nothing, PSDMatrix(K.C.R.blocks[i]))
        _C_DxD = C_DxD.blocks[i]
        _C_3DxD = C_3DxD.blocks[i]
        marginalize_cov!(_Σ_out, _Σ_curr, _K; C_DxD=_C_DxD, C_3DxD=_C_3DxD)
    end
    return Σ_out
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
    KT1<:AffineNormalKernel{<:AbstractMatrix,<:AbstractVecOrMat,<:PSDMatrix},
    KT2<:AffineNormalKernel{<:AbstractMatrix,<:Any,<:PSDMatrix},
}
    # @assert Matrix(UpperTriangular(xpred.Σ.R)) == Matrix(xpred.Σ.R)

    A, _, Q = K
    G, b, Λ = Kout

    D = output_dim = size(G, 1)

    # G = Matrix(x.Σ) * A' / Matrix(xpred.Σ)
    _matmul!(C_DxD, x.Σ.R, A')
    _matmul!(G, x.Σ.R', C_DxD)
    rdiv!(G, Cholesky(xpred.Σ.R, 'U', 0))

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
        _matmul!(view(Λ.R, D+1:2D, 1:D), Q.R, G')
    else
        apply_diffusion!(PSDMatrix(C_DxD), Q, diffusion)
        _matmul!(view(Λ.R, D+1:2D, 1:D), C_DxD, G')
    end

    return Kout
end

function compute_backward_kernel!(
    Kout::KT1,
    xpred::SRGaussian{T,<:IsometricKroneckerProduct},
    x::SRGaussian{T,<:IsometricKroneckerProduct},
    K::KT2;
    C_DxD::AbstractMatrix,
    diffusion::Union{Number,Diagonal}=1,
) where {
    T,
    KT1<:AffineNormalKernel{
        <:IsometricKroneckerProduct,
        <:AbstractVector,
        <:PSDMatrix{T,<:IsometricKroneckerProduct},
    },
    KT2<:AffineNormalKernel{
        <:IsometricKroneckerProduct,
        <:Any,
        <:PSDMatrix{T,<:IsometricKroneckerProduct},
    },
}
    D = length(x.μ)  # full_state_dim
    d = K.A.ldim     # ode_dimension_dim
    Q = D ÷ d        # n_derivatives_dim
    _Kout =
        AffineNormalKernel(Kout.A.B, reshape_no_alloc(Kout.b, Q, d), PSDMatrix(Kout.C.R.B))
    _x_pred = Gaussian(reshape_no_alloc(xpred.μ, Q, d), PSDMatrix(xpred.Σ.R.B))
    _x = Gaussian(reshape_no_alloc(x.μ, Q, d), PSDMatrix(x.Σ.R.B))
    _K = AffineNormalKernel(K.A.B, reshape_no_alloc(K.b, Q, d), PSDMatrix(K.C.R.B))
    _C_DxD = C_DxD.B
    _diffusion =
        diffusion isa Number ? diffusion :
        diffusion isa IsometricKroneckerProduct ? diffusion.B : diffusion

    return compute_backward_kernel!(
        _Kout, _x_pred, _x, _K; C_DxD=_C_DxD, diffusion=_diffusion)
end

function compute_backward_kernel!(
    Kout::KT1,
    xpred::SRGaussian{T,<:BlockDiag},
    x::SRGaussian{T,<:BlockDiag},
    K::KT2;
    C_DxD::AbstractMatrix,
    diffusion::Union{Number,Diagonal}=1,
) where {
    T,
    KT1<:AffineNormalKernel{
        <:BlockDiag,
        <:AbstractVector,
        <:PSDMatrix{T,<:BlockDiag},
    },
    KT2<:AffineNormalKernel{
        <:BlockDiag,
        <:Any,
        <:PSDMatrix{T,<:BlockDiag},
    },
}
    d = length(blocks(xpred.Σ.R))
    q = size(blocks(xpred.Σ.R)[1], 1) - 1

    @simd ivdep for i in eachindex(blocks(xpred.Σ.R))
        _Kout = AffineNormalKernel(
            Kout.A.blocks[i],
            view(Kout.b, (i-1)*(q+1)+1:i*(q+1)),
            PSDMatrix(Kout.C.R.blocks[i]),
        )
        _xpred = Gaussian(
            view(xpred.μ, (i-1)*(q+1)+1:i*(q+1)),
            PSDMatrix(xpred.Σ.R.blocks[i]),
        )
        _x = Gaussian(
            view(x.μ, (i-1)*(q+1)+1:i*(q+1)),
            PSDMatrix(x.Σ.R.blocks[i]),
        )
        _K = AffineNormalKernel(
            K.A.blocks[i],
            ismissing(K.b) ? missing : view(K.b, (i-1)*(q+1)+1:i*(q+1)),
            PSDMatrix(K.C.R.blocks[i]),
        )
        _C_DxD = C_DxD.blocks[i]
        _diffusion = diffusion isa Number ? diffusion : diffusion[i]
        compute_backward_kernel!(
            _Kout, _xpred, _x, _K, C_DxD=_C_DxD, diffusion=_diffusion
        )
    end
    return Kout
end
