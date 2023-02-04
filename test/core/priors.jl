using ProbNumDiffEq
import ProbNumDiffEq as PNDE
using Test
using LinearAlgebra

import ODEProblemLibrary: prob_ode_lotkavolterra

h = rand()
σ = rand()

@testset "Test vanilla (ie. non-preconditioned) IBM (d=2,q=2)" begin
    d, q = 2, 2

    prior = PNDE.IWP(d, q)
    Ah, Qh = PNDE.discretize(prior, h)
    Qh = PNDE.apply_diffusion(Qh, σ^2)

    AH_22_IBM = [
        1 h h^2/2 0 0 0
        0 1 h 0 0 0
        0 0 1 0 0 0
        0 0 0 1 h h^2/2
        0 0 0 0 1 h
        0 0 0 0 0 1
    ]
    @test AH_22_IBM ≈ Ah

    QH_22_IBM =
        σ^2 .* [
            h^5/20 h^4/8 h^3/6 0 0 0
            h^4/8 h^3/3 h^2/2 0 0 0
            h^3/6 h^2/2 h 0 0 0
            0 0 0 h^5/20 h^4/8 h^3/6
            0 0 0 h^4/8 h^3/3 h^2/2
            0 0 0 h^3/6 h^2/2 h
        ]
    @test QH_22_IBM ≈ Matrix(Qh)
end

@testset "Test IBM with preconditioning (d=2,q=2)" begin
    d, q = 2, 2

    prior = PNDE.IWP(d, q)
    A, Q = PNDE.preconditioned_discretize(prior)
    Qh = PNDE.apply_diffusion(Q, σ^2)

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

    @test AH_22_PRE ≈ Matrix(A)
    @test QH_22_PRE ≈ Matrix(Qh)
end

@testset "Verify correct prior dim" begin
    prob = prob_ode_lotkavolterra
    d = length(prob.u0)
    for q in 1:5
        integ = init(prob, EK0(order=q))
        @test length(integ.cache.x.μ) == d * (q + 1)
        sol = solve!(integ)
        @test length(integ.cache.x.μ) == d * (q + 1)
        @test length(sol.x_filt[end].μ) == d * (q + 1)
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

    A1, Q1 = PNDE.discretize(sde, h)
    A2, Q2 = PNDE.discretize(prior, h)

    @test A1 ≈ A2
    @test Q1 ≈ Matrix(Q2)
end
