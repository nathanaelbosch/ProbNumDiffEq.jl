using ProbNumDiffEq
import ProbNumDiffEq as PNDE
import ProbNumDiffEq: AffineNormalKernel
using LinearAlgebra, Statistics
using FillArrays
using Test

d_in, d_out = 3, 2
d_out = d_in
D = 3

T = Float64
@testset "T=$T" for T in (Float64, BigFloat)
    @testset "FAC=$_FAC" for _FAC in (
        PNDE.IsometricKroneckerCovariance,
        PNDE.BlockDiagonalCovariance,
        PNDE.DenseCovariance,
    )
        _FAC = PNDE.DenseCovariance
        FAC = _FAC{T}(d_in, D - 1)

        _A = PNDE.RightIsometricKroneckerProduct(D, rand(T, d_out, d_in))
        _b = rand(T, d_out * D)
        _C = PSDMatrix(PNDE.RightIsometricKroneckerProduct(D, rand(T, d_out, d_out)))

        A = PNDE.to_factorized_matrix(FAC, _A)
        for b in (_b, Zeros(T, size(_b)...)),
            C in (PNDE.to_factorized_matrix(FAC, _C), PSDMatrix(Zeros(T, size(_C)...)))

            K = AffineNormalKernel(A, b, C)

            # definition
            x = rand(T, d_in * D)
            @test K(x) == Gaussian(A * x + b, C)

            # base
            @test_nowarn _, _, _ = K
            @test (@test_nowarn similar(K) isa AffineNormalKernel)
            @test copy(K) == AffineNormalKernel(copy(A), copy(b), copy(C))
            _K = similar(K)
            @test copy!(_K, K) == K

            # marginalize
            d_ov = 2 * d_in
            x = Gaussian(rand(T, d_in * D), PSDMatrix(rand(T, d_ov * D, d_in * D)))
            y = @test_nowarn PNDE.marginalize(x, K)
            @test y isa Gaussian
            @test mean(y) ≈ A * mean(x) + b
            @test PNDE.unfactorize(cov(y)) ≈
                  A * PNDE.unfactorize(cov(x)) * A' + PNDE.unfactorize(C)

            # forward-backward marginalization returns the original!
            Kback = @test_nowarn PNDE.compute_backward_kernel(y, x, K)
            xback = @test_nowarn PNDE.marginalize(y, Kback)
            @test mean(xback) ≈ mean(x)
            @test PNDE.unfactorize(cov(xback)) ≈ PNDE.unfactorize(cov(x))

            # marginalize in-place
            y = similar(y)
            cachemat = rand(T, size(cov(x).R, 1) + size(K.C.R, 1), size(A, 1))
            @test_nowarn PNDE.marginalize!(y, x, K; cachemat)
            @test mean(y) ≈ A * mean(x) + b
            @test PNDE.unfactorize(cov(y)) ≈
                  A * PNDE.unfactorize(cov(x)) * A' + PNDE.unfactorize(C)
        end
    end
end
