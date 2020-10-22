using ProbNumODE
using Test
using Plots
using GaussianDistributions
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo


@testset "Solution" begin
    prob = ProbNumODE.remake_prob_with_jac(prob_ode_lotkavoltera)
    sol = solve(prob, EKF1(), adaptive=false, dt=1e-2)

    @test length(sol) > 2
    @test length(sol.t) == length(sol.u)
    @test length(prob.u0) == length(sol.u[end])

    # Destats
    @testset "DEStats" begin
        @test length(sol.t) == sol.destats.naccept + 1
        @test sol.destats.naccept <= sol.destats.nf
    end

    @testset "Hit the provided tspan" begin
        @test sol.t[1] ≈ prob.tspan[1]
        @test sol.t[end] ≈ prob.tspan[2]
    end

    @testset "Prob and non-prob u" begin
        @test sol.u == sol.pu.μ
    end

    @testset "Call on known t" begin
        @test sol.(sol.t) == sol.u
        u0 = sol(prob.tspan[1])
        @test prob.u0 == u0

        pu0 = sol.p(prob.tspan[1])
        # @test pu0.Σ ≈ eps(eltype(pu0.Σ))*I
        @test pu0.Σ == zeros(size(pu0.Σ))
    end

    # Interpolation
    @testset "Dense Solution" begin
        t0 = prob.tspan[1]
        t1, t2 = t0 + 1e-2, t0 + 2e-2

        u0, u1, u2 = sol(t0), sol(t1), sol(t2)
        @test norm.(u0 - u1) < norm.(u0 - u2)

        pu1, pu2 = sol.p(t0), sol.p(t1)
        @test all(diag(pu1.Σ) .< diag(pu2.Σ))

        @test sol(t0:1e-3:t1) isa DiffEqBase.DiffEqArray
    end

    @testset "GaussianODEFilterPosterior" begin
        t = prob.tspan[1] + 1e-2
        @test sol.p(t) isa Gaussian
        @test sol.p[1] == sol.pu[1]
        @test sol.p[length(sol.pu)] == sol.pu[end]
    end

    @testset "GaussianFilteringPosterior" begin
        filtpost = sol.p.filtering_posterior
        t = prob.tspan[1] + 1e-2
        @test filtpost(t) isa Gaussian
        @test filtpost[1] == sol.x[1]
        @test filtpost[length(sol.x)] == sol.x[end]
    end
end
