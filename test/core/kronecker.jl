using Test
using LinearAlgebra
using Random
import ProbNumDiffEq as PNDE

d = 2
q = 2

@testset "$T" for T in (Float64, BigFloat)
    R1 = randn(T, q + 1, q + 1)
    K1 = PNDE.IsometricKroneckerProduct(d, R1)
    M1 = Matrix(K1)
    R2 = randn(T, q + 1, q + 1)
    K2 = PNDE.IsometricKroneckerProduct(d, R2)
    M2 = Matrix(K2)

    # Matrix-Matrix Operations
    @test K1 * K2 ≈ M1 * M2
    @test K1 * K2 isa PNDE.IsometricKroneckerProduct
    @test K1 + K2 ≈ M1 + M2
    @test K1 + K2 isa PNDE.IsometricKroneckerProduct

    # DimensionMismatch
    X = PNDE.IsometricKroneckerProduct(d, randn(T, 1, 1))
    @test_throws DimensionMismatch X + K1
    @test_throws DimensionMismatch K1 + X
    @test_throws DimensionMismatch X * K1
    @test_throws DimensionMismatch K1 * X

    # IsometricKroneckerProduct from Vector
    R4 = randn(T, q + 1)
    K4 = PNDE.IsometricKroneckerProduct(d, R4)
    M4 = Matrix(K4)
    @test K1 * K4 ≈ M1 * M4
    @test_throws DimensionMismatch K1 + K4

    # UniformScaling
    @test I + K1 ≈ I + M1
    @test I + K1 isa PNDE.IsometricKroneckerProduct
    @test K1 + I ≈ M1 + I
    @test K1 + I isa PNDE.IsometricKroneckerProduct
    @test I - K1 ≈ I - M1
    @test I - K1 isa PNDE.IsometricKroneckerProduct
    @test K1 - I ≈ M1 - I
    @test K1 - I isa PNDE.IsometricKroneckerProduct

    # Other LinearAlgebra
    @test K1' ≈ M1'
    @test K1' isa PNDE.IsometricKroneckerProduct
    @test inv(K1) ≈ inv(M1)
    @test inv(K1) isa PNDE.IsometricKroneckerProduct
    @test det(K1) ≈ det(M1)
    @test K1 / K2 ≈ M1 / M2
    @test K1 / K2 isa PNDE.IsometricKroneckerProduct
    @test K1 \ K2 ≈ M1 \ M2
    @test K1 \ K2 isa PNDE.IsometricKroneckerProduct

    # Base
    K3 = PNDE.IsometricKroneckerProduct(d, copy(R2))
    M3 = Matrix(K3)
    @test similar(K1) isa PNDE.IsometricKroneckerProduct
    @test copy(K1) isa PNDE.IsometricKroneckerProduct
    @test copy!(K3, K1) isa PNDE.IsometricKroneckerProduct
    @test K3 == K1
    @test size(K1) == size(M1)

    # Base
    @test one(K1) isa PNDE.IsometricKroneckerProduct
    @test isone(one(K1).B)
    @test zero(K1) isa PNDE.IsometricKroneckerProduct
    @test iszero(zero(K1).B)

    # Matrix-Scalar
    α = 2.0
    @test α * K1 ≈ α * M1
    @test α * K1 isa PNDE.IsometricKroneckerProduct
    @test K1 * α ≈ α * M1
    @test K1 * α isa PNDE.IsometricKroneckerProduct

    # In-place Matrix-Matrix Multiplication
    β = -0.5
    @test mul!(K3, K1, K2) ≈ mul!(M3, M1, M2)
    @test mul!(K3, K1, K2, α, β) ≈ mul!(M3, M1, M2, α, β)

    # Fast In-place Matrix-Matrix Multiplication
    @test PNDE._matmul!(K3, K1, K2) ≈ PNDE._matmul!(M3, M1, M2)
    @test PNDE._matmul!(K3, K1, K2, α, β) ≈ PNDE._matmul!(M3, M1, M2, α, β)

    # DimensionMismatch
    @test_throws DimensionMismatch mul!(X, K1, K2)
    @test_throws DimensionMismatch mul!(X, K1, K2, α, β)
    @test_throws DimensionMismatch PNDE._matmul!(X, K1, K2)
    @test_throws DimensionMismatch PNDE._matmul!(X, K1, K2, α, β)

    # Kronecker-trick
    v = rand(T, d * (q + 1))
    A = rand(T, d * (q + 1), d * (q + 1))
    @test K1 * v ≈ M1 * v
    @test K1 * A ≈ M1 * A
    @test mul!(copy(v), K1, v) ≈ mul!(copy(v), M1, v)
    @test mul!(copy(A), K1, A) ≈ mul!(copy(A), M1, A)
    @test mul!(copy(v), K1, v, α, β) ≈ mul!(copy(v), M1, v, α, β)
    @test mul!(copy(A), K1, A, α, β) ≈ mul!(copy(A), M1, A, α, β)

    @test mul!(copy(A'), copy(A'), K1') ≈ mul!(copy(A'), A', M1')
    @test mul!(copy(A'), copy(A'), K1', α, β) ≈ mul!(copy(A'), A', M1', α, β)

    @test PNDE._matmul!(copy(v), K1, v) ≈ PNDE._matmul!(copy(v), M1, v)
    @test PNDE._matmul!(copy(A), K1, A) ≈ PNDE._matmul!(copy(A), M1, A)
    @test PNDE._matmul!(copy(v), K1, v, α, β) ≈ PNDE._matmul!(copy(v), M1, v, α, β)
    @test PNDE._matmul!(copy(A), K1, A, α, β) ≈ PNDE._matmul!(copy(A), M1, A, α, β)

    if T == Float64
        # Octavian has issues
        @test_broken PNDE._matmul!(copy(A'), copy(A'), K1') ≈
                     PNDE._matmul!(copy(A'), A', M1')
        @test_broken PNDE._matmul!(copy(A'), copy(A'), K1', α, β) ≈
                     PNDE._matmul!(copy(A'), A', M1', α, β)
    else
        # Uses LinearAlgebra
        @test PNDE._matmul!(copy(A'), copy(A'), K1') ≈ PNDE._matmul!(copy(A'), A', M1')
        @test PNDE._matmul!(copy(A'), copy(A'), K1', α, β) ≈
              PNDE._matmul!(copy(A'), A', M1', α, β)
    end
    # But it always works if all matrices are actual adjoints
    @test PNDE._matmul!(copy(A)', A', K1') ≈ PNDE._matmul!(copy(A'), A', M1')
    @test PNDE._matmul!(copy(A)', A', K1', α, β) ≈ PNDE._matmul!(copy(A'), A', M1', α, β)
end
