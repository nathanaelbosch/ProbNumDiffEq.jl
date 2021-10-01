"""
    predict(x, A, Q)

PREDICT step in Kalman filtering for linear dynamics models:
```math
m_{n+1}^P = A*m_n
C_{n+1}^P = A*C_n*A^T + Q
```

This function provides a very simple PREDICT implementation.
In the solvers, we recommend to use the non-allocating [`predict!`](@ref).
"""
predict(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) =
    Gaussian(predict_mean(x, A), predict_cov(x, A, Q))
predict_mean(x::Gaussian, A::AbstractMatrix) = A*x.μ
predict_cov(x::Gaussian, A::AbstractMatrix, Q::AbstractMatrix) = A*x.Σ*A' + Q


"""Square-root implementation of [`predict_cov!`](@ref)."""
predict_cov(x::SRGaussian, A::AbstractMatrix, Q::SRMatrix) =
    SRMatrix(qr([A * x.Σ.squareroot Q.squareroot]').R')


"""
    predict!(x_out, x_curr, Ah, Qh, cachemat)

PREDICT step in Kalman filtering for linear dynamics models.
In-place implementation of [`predict`](@ref), saving the result in `x_out`.

```math
m_{n+1}^P = A(h)*m_n
P_{n+1}^P = A(h)*P_n*A(h) + Q(h)
```

See also: [`predict`](@ref)
"""
function predict!(x_out::SRGaussian, x_curr::SRGaussian, Ah::AbstractMatrix, Qh::SRMatrix,
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
