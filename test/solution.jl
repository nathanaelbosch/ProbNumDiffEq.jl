using ODEFilters
using Test
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo


@testset "Solution" begin
    prob = ODEFilters.remake_prob_with_jac(prob_ode_lotkavoltera)
    sol = solve(prob, EK1())

    @test length(sol) > 2
    @test length(sol.t) == length(sol.u)
    @test length(prob.u0) == length(sol.u[end])

    # Destats
    @testset "DEStats" begin
        @test length(sol.t) == sol.destats.naccept + 1
        @test sol.destats.naccept <= sol.destats.nf
    end

    @testset "Hit the provided tspan" begin
        @test sol.t[1] == prob.tspan[1]
        @test sol.t[end] == prob.tspan[2]
    end

    @testset "Prob and non-prob u" begin
        @test sol.u == sol.pu.μ
    end

    @testset "Call on known t" begin
        @test sol(sol.t).u == sol.pu
    end

    @testset "Correct initial values" begin
        @test sol.pu[1].μ == prob.u0
        @test iszero(sol.pu[1].Σ)
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
    end

    # Sampling
    @testset "Solution Sampling" begin
        samples = ODEFilters.sample(sol, 10)

        @test samples isa Array

        m, n, o = size(samples)
        @test m == length(sol)
        @test n == length(sol.u[1])
        @test o == 10

        u = ODEFilters.stack(sol.u)
        stds = sqrt.(ODEFilters.stack(diag.(sol.pu.Σ)))
        outlier_count = sum(abs.(u .- samples) .> 3stds)
        @assert outlier_count < 0.05 * m * n * o

        # Dense sampling
        dense_samples, dense_times = ODEFilters.dense_sample(sol, 10)
        m, n, o = size(dense_samples)
        @test m == length(dense_times)
        @test n == length(sol.u[1])
        @test o == 10
    end
end
