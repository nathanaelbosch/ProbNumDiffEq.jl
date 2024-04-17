using ProbNumDiffEq
import ProbNumDiffEq as PNDE
import ProbNumDiffEq: SelectionMatrix
using LinearAlgebra
using Test

d_out = 2
d_in = 3
d_lat = 4

@testset "T=$T" for T in (Float64, BigFloat)
    E = SelectionMatrix([2, 1], d_in)
    M = Matrix(E)

    v = rand(d_in)
    @test E * v == M * v
    X = rand(d_in, d_lat)
    @test E * X == M * X
    @test X' * E' == X' * M'
    Xt = Matrix(X')
    @test Xt * E' == Xt * M'

    v = rand(d_out)
    @test E' * v == M' * v
    X = rand(d_out, d_lat)
    @test E' * X == M' * X
end
