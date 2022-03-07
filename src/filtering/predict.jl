"""
    predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix)

Prediction step in Kalman filtering for linear dynamics models.

Given a Gaussian ``x = \\mathcal{N}(μ, Σ)``, compute and return
``\\mathcal{N}(A μ, A Σ A^T + Q)``.

See also the non-allocating square-root version [`predict!`](@ref).
"""
predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) =
    Gaussian(predict_mean(x, A), predict_cov(x, A, Q))
predict_mean(x::Gaussian, A::AbstractMatrix) = A * x.μ
predict_cov(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) = A * x.Σ * A' + Q
predict_cov(x::SRGaussian, A::AbstractMatrix, Q::SRMatrix) =
    SRMatrix(qr([A * x.Σ.squareroot Q.squareroot]').R')

"""
    predict!(x_out, x_curr, Ah, Qh, cachemat)

In-place and square-root implementation of [`predict`](@ref)
which saves the result into `x_out`.

Only works with `ProbNumDiffEq.SquarerootMatrix` types as `Ah`, `Qh`, and in the
covariances of `x_curr` and `x_out` (both of type `Gaussian`).
To prevent allocations, a cache matrix `cachemat` of size ``D \\times 2D``
(where ``D \\times D`` is the size of `Ah` and `Qh`) needs to be passed.

See also: [`predict`](@ref).
"""
function predict!(
    x_out::SRGaussian,
    x_curr::SRGaussian,
    Ah::AbstractMatrix,
    Qh::SRMatrix,
    cachemat::SRMatrix,
    diffusion=1,
)
    predict_mean!(x_out, x_curr, Ah)
    predict_cov!(x_out, x_curr, Ah, Qh, cachemat, diffusion)
    return x_out
end

function predict_mean!(x_out::Gaussian, x_curr::Gaussian, Ah::AbstractMatrix)
    mul!(x_out.μ, Ah, x_curr.μ)
    return x_out.μ
end

function predict_cov!(
    x_out::SRGaussian,
    x_curr::SRGaussian,
    Ah::AbstractMatrix,
    Qh::SRMatrix,
    cachemat::SRMatrix,
    diffusion=1,
)
    M, L = cachemat.mat, cachemat.squareroot
    D, D = size(Qh.mat)

    _matmul!(view(L, 1:D, 1:D), Ah, x_curr.Σ.squareroot)
    _matmul!(view(L, 1:D, D+1:2D), sqrt.(diffusion), Qh.squareroot)
    _matmul!(M, L, L')
    chol = cholesky!(Symmetric(M), check=false)

    QL =
        issuccess(chol) ? Matrix(chol.U)' :
        eltype(L) <: Union{Float16,Float32,Float64} ? lq!(L).L : qr(L').R'
    copy!(x_out.Σ.squareroot, QL)
    _matmul!(x_out.Σ.mat, QL, QL')
    return x_out.Σ
end
