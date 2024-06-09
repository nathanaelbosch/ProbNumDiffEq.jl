using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using Statistics, LinearAlgebra
using Test

d = 3

function test_base(G)
    m, C = G.μ, G.Σ
    @test G == Gaussian(m, C)
    @test G ≈ Gaussian(m, C)

    @test copy(G) == G
    if G.μ isa AbstractVector
        @test_nowarn similar(G)
    end
    @test length(G) == length(G.μ)
    @test size(G) == size(G.μ)
    @test eltype(typeof(G)) == typeof(G)
end

function test_stats(G::Gaussian{TM,TC}) where {TM,TC}
    @test mean(G) == G.μ
    @test cov(G) == G.Σ
    if G.μ isa Number
        @test var(G) == G.Σ
    else
        @test var(G) == diag(G.Σ)
    end
    @test std(G) == sqrt.(var(G))

    s = @test_nowarn rand(G)
    @test s isa TM
    @test PNDE.pdf(G, s) isa eltype(TM)

    @test typeof(G + s) == typeof(G)
    @test typeof(s + G) == typeof(G)
    @test typeof(G - s) == typeof(G)
    M = rand(size(G.Σ)...)
    @test typeof(M*G) == typeof(G)
end

@testset "T=$T" for T in (Float64, BigFloat)
    # scalars
    @testset "scalar-valued" begin
        G = Gaussian(rand(T), rand(T))
        test_base(G)
        test_stats(G)
    end

    @testset "vector-valued" begin
        m = rand(T, d)
        C = PSDMatrix(rand(T, d, d))
        G = Gaussian(m, Matrix(C))
        test_base(G)
        test_stats(G)
    end
end
