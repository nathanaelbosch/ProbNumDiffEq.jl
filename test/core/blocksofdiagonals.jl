using ProbNumDiffEq
import ProbNumDiffEq as PNDE
import ProbNumDiffEq: BlocksOfDiagonals, _matmul!
using LinearAlgebra
using BlockDiagonals
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

    @test Matrix(BlockDiagonal(A)) == AM
    @test Matrix(BlockDiagonal(B)) == BM
    @test Matrix(BlockDiagonal(C)) == CM

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

    @test_throws ErrorException view(A, 1:2, 1:2)

    tttm(copy!(copy(A), Diagonal(A)))

    @test tttm(A + A) ≈ AM + AM
    @test tttm(A - A) ≈ AM - AM

    _A = copy(A)
    @test tttm(PNDE.add!(_A, A)) == AM + AM
    @test Matrix(_A) == AM + AM
end
