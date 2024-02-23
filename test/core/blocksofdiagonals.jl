using ProbNumDiffEq
import ProbNumDiffEq as PNDE
import ProbNumDiffEq: BlocksOfDiagonals, _matmul!, nblocks
using LinearAlgebra
import BlockArrays
using Test

d1, d2 = 2, 3
D = d1 * d2

@testset "T=$T" for T in (Float64, BigFloat)
    A = BlocksOfDiagonals([randn(T, d1, d1) for _ in 1:d2])
    B = BlocksOfDiagonals([randn(T, d1, d1) for _ in 1:d2])
    C = BlocksOfDiagonals([randn(T, d1, d1) for _ in 1:d2])
    alpha = rand(T)
    beta = rand(T)

    AM, BM, CM = @test_nowarn Matrix.((A, B, C))

    @test Matrix(BlockArrays.BlockArray(A)) == AM
    @test Matrix(BlockArrays.BlockArray(B)) == BM
    @test Matrix(BlockArrays.BlockArray(C)) == CM

    _A = @test_nowarn copy(A)
    @test _A isa BlocksOfDiagonals

    _B = @test_nowarn copy!(_A, B)
    @test _B === _A
    @test _B == B

    _A = @test_nowarn similar(A)
    @test _A isa BlocksOfDiagonals
    @test size(_A) == size(A)

    _Z = @test_nowarn zero(A)
    @test _Z isa BlocksOfDiagonals
    @test size(_Z) == size(A)
    @test all(_Z .== 0)

    function tttm(M) # quick type test and to matrix
        @test M isa BlocksOfDiagonals
        return Matrix(M)
    end

    @test tttm(mul!(C, A, B)) ≈ mul!(CM, AM, BM)
    @test tttm(mul!(C, A', B)) ≈ mul!(CM, AM', BM)
    @test tttm(mul!(C, A, B')) ≈ mul!(CM, AM, BM')
    @test tttm(_matmul!(C, A, B)) ≈ _matmul!(CM, AM, BM)
    @test tttm(_matmul!(C, A', B)) ≈ _matmul!(CM, AM', BM)
    @test tttm(_matmul!(C, A, B')) ≈ _matmul!(CM, AM, BM')

    @test tttm(mul!(C, A, B, alpha, beta)) ≈ mul!(CM, AM, BM, alpha, beta)
    @test tttm(_matmul!(C, A, B, alpha, beta)) ≈ _matmul!(CM, AM, BM, alpha, beta)

    @test tttm(A * B) ≈ AM * BM
    @test tttm(A' * B) ≈ AM' * BM
    @test tttm(A * B') ≈ AM * BM'

    @test tttm(A * alpha) ≈ AM * alpha
    @test tttm(alpha * A) ≈ alpha * AM
    @test tttm(A * (alpha * I)) ≈ AM * alpha
    @test tttm((alpha * I) * A) ≈ alpha * AM
    @test tttm(rmul!(copy(A), alpha)) ≈ alpha * AM
    @test tttm(mul!(_A, alpha, A)) ≈ alpha * AM
    @test tttm(mul!(_A, A, alpha)) ≈ alpha * AM
    @test tttm(_matmul!(_A, alpha, A)) ≈ alpha * AM
    @test tttm(_matmul!(_A, A, alpha)) ≈ alpha * AM

    @test tttm((alpha * I(D)) * A) ≈ alpha * AM
    @test tttm(A * (alpha * I(D))) ≈ AM * alpha
    @test tttm(mul!(_A, A, alpha * I(D))) ≈ alpha * AM
    @test tttm(mul!(_A, alpha * I(D), A)) ≈ alpha * AM
    @test tttm(_matmul!(_A, A, alpha * I(D))) ≈ alpha * AM
    @test tttm(_matmul!(_A, alpha * I(D), A)) ≈ alpha * AM

    # Actual Diagonals
    DI = Diagonal(rand(T, D))
    @test tttm(DI * A) ≈ DI * AM
    @test tttm(A * DI) ≈ AM * DI
    @test tttm(mul!(copy(A), DI, A)) ≈ DI * AM
    @test tttm(mul!(copy(A), A, DI)) ≈ AM * DI
    @test tttm(_matmul!(copy(A), DI, A)) ≈ DI * AM
    @test tttm(_matmul!(copy(A), A, DI)) ≈ AM * DI
    @test tttm(mul!(copy(A), DI, A, alpha, beta)) ≈ mul!(copy(AM), DI, AM, alpha, beta)
    @test tttm(mul!(copy(A), A, DI, alpha, beta)) ≈ mul!(copy(AM), AM, DI, alpha, beta)
    @test tttm(_matmul!(copy(A), DI, A, alpha, beta)) ≈ mul!(copy(AM), DI, AM, alpha, beta)
    @test tttm(_matmul!(copy(A), A, DI, alpha, beta)) ≈ mul!(copy(AM), AM, DI, alpha, beta)

    @test_throws ErrorException view(A, 1:2, 1:2)

    _D = Diagonal(rand(T, D))
    @test tttm(copy!(copy(A), _D)) == _D

    @test tttm(A + A) ≈ AM + AM
    @test tttm(A - A) ≈ AM - AM

    _A = copy(A)
    @test tttm(PNDE.add!(_A, A)) == AM + AM
    @test Matrix(_A) == AM + AM

    # Matrix-Vector Products
    v = rand(T, size(A, 2))
    @test A * v == AM * v
    x = rand(T, size(A, 1))
    @test mul!(x, A, v) == mul!(x, AM, v)
    @test mul!(x, A, v, alpha, beta) == mul!(x, AM, v, alpha, beta)
    @test _matmul!(x, A, v) == _matmul!(x, AM, v)
    @test _matmul!(x, A, v, alpha, beta) == _matmul!(x, AM, v, alpha, beta)

    @test tttm([A; B]) == [AM; BM]
    @test tttm([A B]) == [AM BM]
    @test_broken [A B; B A] isa PNDE.BlocksOfDiagonals
end
