using ProbNumDiffEq
using Test
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using ODEProblemLibrary: prob_ode_lotkavolterra

@testset "Solution" begin
    prob = prob_ode_lotkavolterra
    sol = solve(prob, EK1())

    @test length(sol) > 2
    @test length(sol.t) == length(sol.u)
    @test length(prob.u0) == length(sol.u[end])

    # Stats
    @testset "Stats" begin
        @test length(sol.t) == sol.stats.naccept + 1
        @test sol.stats.naccept <= sol.stats.nf
    end

    @testset "Hit the provided tspan" begin
        @test sol.t[1] == prob.tspan[1]
        @test sol.t[end] == prob.tspan[2]
    end

    @testset "Prob and non-prob u" begin
        @test sol.u == sol.pu.μ
    end

    @testset "Prob u properties" begin
        @test_nowarn mean(sol.pu[1])
        @test_nowarn cov(sol.pu[1])
        @test_nowarn var(sol.pu[1])
        @test_nowarn std(sol.pu[1])
    end

    @testset "Call on known t" begin
        @test sol(sol.t).u == sol.pu
    end

    @testset "Correct initial values" begin
        @test sol.pu[1].μ == prob.u0
        @test iszero(sol.pu[1].Σ.R)
    end

    # Interpolation
    @testset "Dense Solution" begin
        t0 = prob.tspan[1]
        t1, t2 = t0 + 1e-2, t0 + 2e-2

        u0, u1, u2 = sol(t0), sol(t1), sol(t2)
        @test norm.(u0.μ - u1.μ) < norm.(u0.μ - u2.μ)

        @test all(diag(u1.Σ) .< diag(u2.Σ))

        @test sol.(t0:1e-3:t1) isa Array{Gaussian{T,S}} where {T,S}
        @test sol(t0:1e-3:t1).u isa StructArray{Gaussian{T,S}} where {T,S}

        @test_throws ErrorException sol(t0 - 1e-2)
    end

    # Sampling
    @testset "Solution Sampling" begin
        n_samples = 2

        samples = ProbNumDiffEq.sample(sol, n_samples)

        @test samples isa Array

        m, n, o = size(samples)
        @test m == length(sol)
        @test n == length(sol.u[1])
        @test o == n_samples

        # Dense sampling
        dense_samples, dense_times = ProbNumDiffEq.dense_sample(sol, n_samples)
        m, n, o = size(dense_samples)
        @test m == length(dense_times)
        @test n == length(sol.u[1])
        @test o == n_samples
    end

    @testset "Sampling states from the solution" begin
        n_samples = 2

        samples = ProbNumDiffEq.sample_states(sol, n_samples)

        @test samples isa Array

        m, n, o = size(samples)
        @test m == length(sol)
        @test n == length(sol.u[1]) * (sol.cache.q + 1)
        @test o == n_samples

        # Dense sampling
        dense_samples, dense_times = ProbNumDiffEq.dense_sample_states(sol, n_samples)
        m, n, o = size(dense_samples)
        @test m == length(dense_times)
        @test n == length(sol.u[1]) * (sol.cache.q + 1)
        @test o == n_samples
    end

    @testset "Plotting" begin
        @test_nowarn plot(sol)
        @test_nowarn plot(sol, denseplot=false)
        @test_nowarn plot(sol, vars=(1, 2))
        @test_nowarn plot(sol, vars=(1, 1, 2))
        @test_nowarn plot(sol, tspan=prob.tspan)
    end

    @testset "Mean Solution" begin
        msol = mean(sol)
        @test_nowarn msol(prob.tspan[1])
        @test_nowarn msol(sol.t[1:2])
        @test_nowarn msol
        @test_nowarn plot(msol)
    end
end
