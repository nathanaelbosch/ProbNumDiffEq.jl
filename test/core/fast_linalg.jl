using ProbNumDiffEq
import ProbNumDiffEq: _matmul!
using LinearAlgebra
using Test

@testset "T=$T" for T in (Float64, BigFloat)
    A = rand(T, 2, 3)
    B = rand(T, 3, 4)
    C = rand(T, 2, 4)
    alpha = rand(T)
    beta = rand(T)

    @test _matmul!(C, A, B) == mul!(C, A, B)
    @test _matmul!(C, A, B, alpha, beta) == mul!(C, A, B, alpha, beta)

    _B = copy(B)
    @test _matmul!(_B, alpha, B) == mul!(_B, alpha, B)
    @test _matmul!(_B, B, beta) == mul!(_B, B, beta)

    # Diagonals
    D = Diagonal(rand(T, size(B, 1)))
    @test _matmul!(_B, D, B) == mul!(_B, D, B)
    @test _matmul!(_B, D, B, alpha, beta) == mul!(_B, D, B, alpha, beta)
    D = Diagonal(rand(T, size(B, 2)))
    @test _matmul!(_B, B, D) == mul!(_B, B, D)
    @test _matmul!(_B, B, D, alpha, beta) == mul!(_B, B, D, alpha, beta)
    CD, D1, D2 = rand(T, 3, 3), Diagonal(rand(T, 3)), Diagonal(rand(T, 3))
    @test _matmul!(CD, D1, D2) == _matmul!(CD, D1, D2)

    # Triangulars
    ASQ, BSQ, CSQ = rand(T, 2, 2), rand(T, 2, 2), rand(T, 2, 2)
    ALT, AUT = LowerTriangular(ASQ), UpperTriangular(ASQ)
    BLT, BUT = LowerTriangular(BSQ), UpperTriangular(BSQ)
    @test _matmul!(CSQ, ALT, BSQ) == mul!(CSQ, ALT, BSQ)
    @test _matmul!(CSQ, AUT, BSQ) == mul!(CSQ, AUT, BSQ)
    @test _matmul!(CSQ, ASQ, BLT) == mul!(CSQ, ASQ, BLT)
    @test _matmul!(CSQ, ASQ, BUT) == mul!(CSQ, ASQ, BUT)
    @test _matmul!(CSQ, ALT, BUT) == mul!(CSQ, ALT, BUT)
    @test _matmul!(CSQ, AUT, BLT) == mul!(CSQ, AUT, BLT)
    @test _matmul!(CSQ, ALT, BLT) == mul!(CSQ, ALT, BLT)
    @test _matmul!(CSQ, AUT, BUT) == mul!(CSQ, AUT, BUT)

    # Adjoints
    AT = Matrix(A')'
    @test _matmul!(C, AT, B) == mul!(C, AT, B)
    BT = Matrix(B')'
    @test _matmul!(C, A, BT) == mul!(C, A, BT)

    # Vectors
    CV, BV = rand(T, 2), rand(T, 3)
    @test _matmul!(CV, A, BV) == mul!(CV, A, BV)
end
