using ProbNumDiffEq
import ProbNumDiffEq: BlockDiag, _matmul!
using LinearAlgebra
using BlockDiagonals
using Test

d1, d2 = 2, 3
@testset "T=$T" for T in (Float64, BigFloat)
    A = BlockDiag([randn(T, d1, d1) for _ in 1:d2])
    B = BlockDiag([randn(T, d1, d1) for _ in 1:d2])
    C = BlockDiag([randn(T, d1, d1) for _ in 1:d2])

    AM, BM, CM = @test_nowarn Matrix.((A, B, C))

    @test Matrix(BlockDiagonal(A)) == AM
    @test Matrix(BlockDiagonal(B)) == BM
    @test Matrix(BlockDiagonal(C)) == CM

    _A = @test_nowarn copy(A)
    @test _A isa BlockDiag

    _B = @test_nowarn copy!(_A, B)
    @test _B === _A
    @test _B == B

    _A = @test_nowarn similar(A)
    @test _A isa BlockDiag
    @test size(_A) == size(A)

    _Z = @test_nowarn zero(A)
    @test _Z isa BlockDiag
    @test size(_Z) == size(A)
    @test all(_Z .== 0)

    function tttm(M) # quick type test and to matrix
        @test M isa BlockDiag
        return Matrix(M)
    end

    @test tttm(mul!(C, A, B)) ≈ mul!(CM, AM, BM)
    @test tttm(mul!(C, A', B)) ≈ mul!(CM, AM', BM)
    @test tttm(mul!(C, A, B')) ≈ mul!(CM, AM, BM')
    @test tttm(_matmul!(C, A, B)) ≈ _matmul!(CM, AM, BM)
    @test tttm(_matmul!(C, A', B)) ≈ _matmul!(CM, AM', BM)
    @test tttm(_matmul!(C, A, B')) ≈ _matmul!(CM, AM, BM')

    @test tttm(A * B) ≈ AM * BM
    @test tttm(A' * B) ≈ AM' * BM
    @test tttm(A * B') ≈ AM * BM'

    a = rand()
    @test tttm(A * a) ≈ AM * a
    @test tttm(a * A) ≈ a * AM
    @test tttm(A * (a * I)) ≈ AM * a
    @test tttm((a * I) * A) ≈ a * AM
    @test tttm(rmul!(copy(A), a)) ≈ a * AM

    @test_throws ErrorException view(A, 1:2, 1:2)

    tttm(copy!(A, Diagonal(A)))
end
