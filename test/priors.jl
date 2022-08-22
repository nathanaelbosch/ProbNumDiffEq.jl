using ProbNumDiffEq
using Test
using LinearAlgebra

import ODEProblemLibrary: prob_ode_lotkavoltera

h = rand()
σ = rand()

@testset "Test vanilla (ie. non-preconditioned) IBM (d=2,q=2)" begin
    d, q = 2, 2

    A!, Q! = ProbNumDiffEq.vanilla_ibm(d, q)
    Ah = diagm(0 => ones(d * (q + 1)))
    Qh = zeros(d * (q + 1), d * (q + 1))
    A!(Ah, h)
    Q!(Qh, h, σ^2)

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
    @test QH_22_IBM ≈ Qh
end

@testset "Test IBM with preconditioning (d=1,q=2)" begin
    d, q = 2, 2

    A, Q = ProbNumDiffEq.ibm(d, q)
    Qh = PSDMatrix(σ * Q.R)

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
    prob = prob_ode_lotkavoltera
    d = length(prob.u0)
    for q in 1:5
        integ = init(prob, EK0(order=q))
        @test length(integ.cache.x.μ) == d * (q + 1)
        sol = solve!(integ)
        @test length(integ.cache.x.μ) == d * (q + 1)
        @test length(sol.x_filt[end].μ) == d * (q + 1)
    end
end
