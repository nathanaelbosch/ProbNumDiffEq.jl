using ProbNumDiffEq: make_transition_matrices!, dim, num_derivatives
using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using Test
using LinearAlgebra
using FiniteHorizonGramians
using Statistics
using Plots
using SimpleUnPack
using FillArrays

h = 0.1

@testset "General prior API" begin
    for prior in (
        IWP(dim=2, num_derivatives=3),
        IOUP(dim=2, num_derivatives=3, rate_parameter=1),
        Matern(dim=2, num_derivatives=3, lengthscale=1),
    )
        d, q = dim(prior), num_derivatives(prior)

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

    σ = 0.1

    prior = PNDE.IWP(dim=d, num_derivatives=q)

    PERM = [1 0 0 0 0 0
            0 0 0 1 0 0
            0 1 0 0 0 0
            0 0 0 0 1 0
            0 0 1 0 0 0
            0 0 0 0 0 1]

    # true sde parameters
    F = [0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 0 0 0 0
        0 0 0 0 1 0
        0 0 0 0 0 1
        0 0 0 0 0 0]
    F = PERM * F * PERM'
    L = [0 0
        0 0
        1 0
        0 0
        0 0
        0 1]
    L = PERM * L

    # true transition matrices
    AH_22_IBM = [1 h h^2/2 0 0 0
        0 1 h 0 0 0
        0 0 1 0 0 0
        0 0 0 1 h h^2/2
        0 0 0 0 1 h
        0 0 0 0 0 1]
    AH_22_IBM = PERM * AH_22_IBM * PERM'

    QH_22_IBM =
        σ^2 .* [h^5/20 h^4/8 h^3/6 0 0 0
            h^4/8 h^3/3 h^2/2 0 0 0
            h^3/6 h^2/2 h 0 0 0
            0 0 0 h^5/20 h^4/8 h^3/6
            0 0 0 h^4/8 h^3/3 h^2/2
            0 0 0 h^3/6 h^2/2 h]
    QH_22_IBM = PERM * QH_22_IBM * PERM'

    # true preconditioned transition matrices
    AH_22_PRE = [
        1 2 1 0 0 0
        0 1 1 0 0 0
        0 0 1 0 0 0
        0 0 0 1 2 1
        0 0 0 0 1 1
        0 0 0 0 0 1
    ]
    AH_22_PRE = PERM * AH_22_PRE * PERM'

    QH_22_PRE =
        σ^2 * [
            1/5 1/4 1/3 0 0 0
            1/4 1/3 1/2 0 0 0
            1/3 1/2 1/1 0 0 0
            0 0 0 1/5 1/4 1/3
            0 0 0 1/4 1/3 1/2
            0 0 0 1/3 1/2 1/1
        ]
    QH_22_PRE = PERM * QH_22_PRE * PERM'

    @testset "Test SDE" begin
        sde = PNDE.to_sde(prior)
        @test sde.F ≈ F
        @test sde.L ≈ L
    end

    @testset "Test vanilla (ie. non-preconditioned)" begin
        Ah, Qh = PNDE.discretize(prior, h)
        @test AH_22_IBM ≈ Ah

        for Γ in (σ^2, σ^2 * Eye(d))
            @test QH_22_IBM ≈ Matrix(PNDE.apply_diffusion(Qh, Γ))
        end
    end

    @testset "Test with preconditioning" begin
        A, Q = PNDE.preconditioned_discretize(prior)
        @test AH_22_PRE ≈ Matrix(A)

        for Γ in (σ^2, σ^2 * Eye(d))
            @test QH_22_PRE ≈ Matrix(PNDE.apply_diffusion(Q, Γ))
        end
    end

    @testset "Test `make_transition_matrices!`" begin
        for FAC in (
            PNDE.IsometricKroneckerCovariance,
            PNDE.BlockDiagonalCovariance,
            )
            A, Q, Ah, Qh, P, PI = PNDE.initialize_transition_matrices(
                FAC{Float64}(d, q), prior, h)

            @test AH_22_PRE ≈ A

            for Γ in (σ^2, σ^2 * Eye(d))
                @test QH_22_PRE ≈ Matrix(PNDE.apply_diffusion(Q, Γ))
            end
            if FAC != PNDE.IsometricKroneckerCovariance
                @test QH_22_PRE ≈ Matrix(PNDE.apply_diffusion(Q, σ^2 * I(d)))
            end

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

            for Γ in (σ^2, σ^2 * Eye(d))
                @test QH_22_IBM ≈ Matrix(PNDE.apply_diffusion(cache.Qh, Γ))
            end
            if FAC != PNDE.IsometricKroneckerCovariance
                @test QH_22_IBM ≈ Matrix(PNDE.apply_diffusion(cache.Qh, σ^2 * I(d)))
            end
        end
    end
end

function test_make_transition_matrices(prior, Atrue, Qtrue)
    d, q = dim(prior), num_derivatives(prior)
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

    prior = PNDE.IOUP(dim=d, num_derivatives=q, rate_parameter=r)

    PERM = [1 0 0 0 0 0
        0 0 0 1 0 0
        0 1 0 0 0 0
        0 0 0 0 1 0
        0 0 1 0 0 0
        0 0 0 0 0 1]

    sde = PNDE.to_sde(prior)
    F = [
        0 1 0 0 0 0
        0 0 1 0 0 0
        0 0 r[1, 1] 0 0 r[1, 2]
        0 0 0 0 1 0
        0 0 0 0 0 1
        0 0 r[2, 1] 0 0 r[2, 2]
    ]
    F = PERM * F * PERM'
    L = [
        0 0
        0 0
        1 0
        0 0
        0 0
        0 1
    ]
    L = PERM * L
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

    prior = PNDE.Matern(dim=d, num_derivatives=q, lengthscale=l)

    PERM = [1 0 0 0 0 0
        0 0 0 1 0 0
        0 1 0 0 0 0
        0 0 0 0 1 0
        0 0 1 0 0 0
        0 0 0 0 0 1]

    sde = PNDE.to_sde(prior)
    F = [
        0 1 0 0 0 0
        0 0 1 0 0 0
        -a(1)*λ^3 -a(2)*λ^2 -a(3)*λ 0 0 0
        0 0 0 0 1 0
        0 0 0 0 0 1
        0 0 0 -a(1)*λ^3 -a(2)*λ^2 -a(3)*λ
    ]
    F = PERM * F * PERM'
    L = [
        0 0
        0 0
        1 0
        0 0
        0 0
        0 1
    ]
    L = PERM * L
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
