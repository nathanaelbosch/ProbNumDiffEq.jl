using ProbNumDiffEq: make_transition_matrices!
using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using Test
using LinearAlgebra

h = 0.1
σ = 0.1

@testset "General prior API" begin
    for prior in (IWP(2, 3), IOUP(2, 3, 1), Matern(2, 3, 1))
        sde = PNDE.to_sde(prior)
        A1, Q1 = PNDE.discretize(sde, h)
        A2, Q2 = PNDE.discretize(prior, h)
        @test A1 ≈ A2
        @test Q1 ≈ Matrix(Q2)
    end
end

@testset "Test IWP (d=2,q=2)" begin
    d, q = 2, 2

    prior = PNDE.IWP(d, q)

    # true sde parameters
    F = [
        0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 0 0 0 0
        0 0 0 0 1 0
        0 0 0 0 0 1
        0 0 0 0 0 0
    ]
    L = [
        0 0
        0 0
        1 0
        0 0
        0 0
        0 1
    ]

    # true transition matrices
    AH_22_IBM = [
        1 h h^2/2 0 0 0
        0 1 h 0 0 0
        0 0 1 0 0 0
        0 0 0 1 h h^2/2
        0 0 0 0 1 h
        0 0 0 0 0 1
    ]

    QH_22_IBM =
        σ^2 .* [
            h^5/20 h^4/8 h^3/6 0 0 0
            h^4/8 h^3/3 h^2/2 0 0 0
            h^3/6 h^2/2 h 0 0 0
            0 0 0 h^5/20 h^4/8 h^3/6
            0 0 0 h^4/8 h^3/3 h^2/2
            0 0 0 h^3/6 h^2/2 h
        ]

    # true preconditioned transition matrices
    AH_22_PRE = [
        1 2 1 0 0 0
        0 1 1 0 0 0
        0 0 1 0 0 0
        0 0 0 1 2 1
        0 0 0 0 1 1
        0 0 0 0 0 1
    ]

    QH_22_PRE =
        σ^2 * [
            1/5 1/4 1/3 0 0 0
            1/4 1/3 1/2 0 0 0
            1/3 1/2 1/1 0 0 0
            0 0 0 1/5 1/4 1/3
            0 0 0 1/4 1/3 1/2
            0 0 0 1/3 1/2 1/1
        ]

    @testset "Test SDE" begin
        sde = PNDE.to_sde(prior)
        @test sde.F ≈ F
        @test sde.L ≈ L
    end

    @testset "Test vanilla (ie. non-preconditioned)" begin
        Ah, Qh = PNDE.discretize(prior, h)
        Qh = PNDE.apply_diffusion(Qh, σ^2)

        @test AH_22_IBM ≈ Ah
        @test QH_22_IBM ≈ Matrix(Qh)
    end

    @testset "Test with preconditioning" begin
        A, Q = PNDE.preconditioned_discretize(prior)
        Qh = PNDE.apply_diffusion(Q, σ^2)

        @test AH_22_PRE ≈ Matrix(A)
        @test QH_22_PRE ≈ Matrix(Qh)
    end

    @testset "Test `make_transition_matrices!`" begin
        struct DummyCache <: PNDE.AbstractODEFilterCache
            d::Any
            q::Any
            A::Any
            Q::Any
            P::Any
            PI::Any
            Ah::Any
            Qh::Any
        end

        A, Q, Ah, Qh, P, PI = PNDE.initialize_transition_matrices(
            PNDE.DenseCovariance{Float64}(d, q), prior, h)
        @test AH_22_PRE ≈ A
        @test QH_22_PRE ≈ Matrix(PNDE.apply_diffusion(Q, σ^2))

        cache = DummyCache(d, q, A, Q, P, PI, Ah, Qh)
        make_transition_matrices!(cache, prior, h)
        @test AH_22_IBM ≈ Ah
        @test QH_22_IBM ≈ Matrix(PNDE.apply_diffusion(cache.Qh, σ^2))
    end
end

@testset "Test IOUP (d=2,q=2)" begin
    d, q = 2, 2
    r = randn(d, d)

    prior = PNDE.IOUP(d, q, r)

    sde = PNDE.to_sde(prior)
    F = [
        0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 r[1, 1] 0 0 r[1, 2]
        0 0 0 0 1 0
        0 0 0 0 0 1
        0 0 r[2, 1] 0 0 r[2, 2]
    ]
    L = [
        0 0
        0 0
        1 0
        0 0
        0 0
        0 1
    ]
    @test sde.F ≈ F
    @test sde.L ≈ L
end

@testset "Test Matern (d=2,q=2)" begin
    d, q = 2, 2
    l = rand()

    ν = q - 1 / 2
    λ = sqrt(2ν) / l
    a(i) = binomial(q + 1, i - 1)

    prior = PNDE.Matern(d, q, l)

    sde = PNDE.to_sde(prior)
    F = [
        0 1 0 0 0 0
        0 0 1 0 0 0
        -a(1)*λ^3 -a(2)*λ^2 -a(3)*λ 0 0 0
        0 0 0 0 1 0
        0 0 0 0 0 1
        0 0 0 -a(1)*λ^3 -a(2)*λ^2 -a(3)*λ
    ]
    L = [
        0 0
        0 0
        1 0
        0 0
        0 0
        0 1
    ]
    @test sde.F ≈ F
    @test sde.L ≈ L
end
