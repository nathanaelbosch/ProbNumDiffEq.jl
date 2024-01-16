using ProbNumDiffEq: make_transition_matrices!, wiener_process_dimension
using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using Test
using LinearAlgebra
using FiniteHorizonGramians
using Statistics
using Plots

h = 0.1
σ = 0.1

@testset "General prior API" begin
    for prior in (IWP(2, 3), IOUP(2, 3, 1), Matern(2, 3, 1))
        d, q = PNDE.wiener_process_dimension(prior), PNDE.num_derivatives(prior)

        sde = PNDE.to_sde(prior)
        A1, Q1 = PNDE.discretize(sde, h)
        A2, Q2 = PNDE.discretize(prior, h)
        @test A1 ≈ A2
        @test Q1 ≈ Q2
        @test Matrix(Q1) ≈ Matrix(Q2)

        A3, Q3 = PNDE.matrix_fraction_decomposition(
            PNDE.drift(sde), PNDE.dispersion(sde), h)
        @test A1 ≈ A3
        @test Matrix(Q1) ≈ Q3

        A4, Q4R = PNDE._discretize_sqrt_with_quadraturetrick(
            PNDE.LTISDE(Matrix(sde.F), Matrix(sde.L)), h)
        @test A1 ≈ A4
        @test Q1.R ≈ Q4R

        ts = 0:0.1:1
        marginals = @test_nowarn PNDE.marginalize(prior, ts)
        @test length(marginals) == length(ts)
        @test marginals[1] isa Gaussian

        N = 3
        samples = @test_nowarn PNDE.sample(prior, ts, N)
        @test length(samples) == length(ts)
        @test size(samples[1]) == (d * (q + 1), N)

        @test_nowarn plot(prior, ts; plot_derivatives=true)
        @test_nowarn plot(prior, ts; plot_derivatives=false)
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
        A, Q, Ah, Qh, P, PI = PNDE.initialize_transition_matrices(
            PNDE.DenseCovariance{Float64}(d, q), prior, h)

        @test AH_22_PRE ≈ A
        @test QH_22_PRE ≈ Matrix(PNDE.apply_diffusion(Q, σ^2))

        cache = (
            d=d,
            q=q,
            A=A,
            Q=Q,
            P=P,
            PI=PI,
            Ah=Ah,
            Qh=Qh,
        )

        make_transition_matrices!(cache, prior, h)
        @test AH_22_IBM ≈ cache.Ah
        @test QH_22_IBM ≈ Matrix(PNDE.apply_diffusion(cache.Qh, σ^2))
    end
end

function test_make_transition_matrices(prior, Atrue, Qtrue)
    @unpack d, q = prior
    @testset "Test `make_transition_matrices!`" begin
        A, Q, Ah, Qh, P, PI = PNDE.initialize_transition_matrices(
            PNDE.DenseCovariance{Float64}(d, q), prior, h)
        F, L = PNDE.to_sde(prior)
        FHG_method = FiniteHorizonGramians.ExpAndGram{eltype(F),13}()
        FHG_cache = FiniteHorizonGramians.alloc_mem(F, L, FHG_method)

        cache = (
            d=d,
            q=q,
            A=A,
            Q=Q,
            P=P,
            PI=PI,
            Ah=Ah,
            Qh=Qh,
            F=F,
            L=L,
            FHG_method=FHG_method,
            FHG_cache=FHG_cache,
            prior=prior,
        )

        make_transition_matrices!(cache, prior, h)

        @test Atrue ≈ cache.Ah
        @test Qtrue ≈ cache.Qh

        @test Atrue ≈ cache.PI * cache.A * cache.P
        @test Qtrue ≈ X_A_Xt(cache.Q, cache.PI)
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

    A1, Q1 = PNDE.discretize(prior, h)
    A2, Q2 = PNDE.discretize(sde, h)
    @test A1 ≈ A2
    @test Q1 ≈ Q2
    @test Matrix(Q1) ≈ Matrix(Q2)

    test_make_transition_matrices(prior, A1, Q1)
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

    A1, Q1 = PNDE.discretize(prior, h)
    A2, Q2 = PNDE.discretize(sde, h)
    @test A1 ≈ A2
    @test Q1 ≈ Q2
    @test Matrix(Q1) ≈ Matrix(Q2)

    test_make_transition_matrices(prior, A1, Q1)

    @testset "Test initial distribution being stationary" begin
        D0 = PNDE.initial_distribution(prior)
        D1 = PNDE.predict(D0, A1, Q1)
        @test mean(D0) ≈ mean(D1)
        @test Matrix(cov(D0)) ≈ Matrix(cov(D1))
    end
end
