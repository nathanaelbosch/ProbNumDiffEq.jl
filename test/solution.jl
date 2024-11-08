using ProbNumDiffEq
using Test
using Plots
using LinearAlgebra
using OrdinaryDiffEq
using Statistics
using ODEProblemLibrary: prob_ode_lotkavolterra

@testset "Solution" begin
    prob1 = prob_ode_lotkavolterra

    prob2 = begin
        du0 = [0.0]
        u0 = [2.0]
        tspan = (0.0, 0.1)
        p = [1e0]
        function vanderpol(du, u, p, t)
            μ = p[1]
            ddu = μ .* ((1 .- u .^ 2) .* du .- u)
            return ddu
        end
        SecondOrderODEProblem(vanderpol, du0, u0, tspan, p)
    end

    @testset "ODE order $ord" for (prob, ord) in ((prob1, 1), (prob2, 2))
        @testset "Alg $Alg" for Alg in (EK0, EK1)
            sol = solve(prob, Alg())

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

                @test sol.(t0:1e-3:t1) isa Array{<:Gaussian}
                @test sol(t0:1e-3:t1).u isa StructArray{<:Gaussian}

                @test_throws ErrorException sol(t0 - 1e-2)

                @testset "Derivatives" begin
                    @test sol.(t0:1e-3:t1, Val{1}) isa Array{<:Gaussian}
                    @test sol(t0:1e-3:t1, Val{1}).u isa StructArray{<:Gaussian}
                    @test_throws ArgumentError sol(1e-3, Val{99})
                end
            end

            # Sampling
            @testset "Solution Sampling" begin
                @testset "Discrete" begin
                    n_samples = 10

                    samples = ProbNumDiffEq.sample(sol, n_samples)

                    @test samples isa Array

                    m, n, o = size(samples)
                    @test m == length(sol)
                    @test n == length(sol.u[1])
                    @test o == n_samples

                    us, es = stack(sol.u), stack(std.(sol.pu))
                    for (interval_width, (low, high)) in (
                        (1, (0.5, 0.8)),
                        (2, (0.8, 0.99)),
                        (3, (0.95, 1)),
                        (4, (0.99, 1)),
                    )
                        percent_in_interval =
                            sum(
                                (
                                sum(
                                    abs.(us .- samples[:, :, i]') .<=
                                    interval_width * es,
                                )
                                for i in 1:n_samples
                            )
                            ) / (m * n * o)
                        @test low <= percent_in_interval <= high
                    end
                end

                @testset "Dense" begin
                    n_samples = 10
                    dense_samples, dense_times =
                        ProbNumDiffEq.dense_sample(sol, n_samples)

                    m, n, o = size(dense_samples)
                    @test m == length(dense_times)
                    @test n == length(sol.u[1])
                    @test o == n_samples

                    pu = sol(dense_times).u
                    us, es = stack(mean.(pu)), stack(std.(pu))
                    for (interval_width, (low, high)) in (
                        (1, (0.5, 0.8)),
                        (2, (0.8, 0.99)),
                        (3, (0.95, 1)),
                        (4, (0.99, 1)),
                    )
                        percent_in_interval =
                            sum(
                                (
                                sum(
                                    abs.(us .- dense_samples[:, :, i]') .<=
                                    interval_width * es,
                                )
                                for i in 1:n_samples
                            )
                            ) / (m * n * o)
                        @test_skip low <= percent_in_interval <= high
                    end
                end
            end

            @testset "Plotting" begin
                @test_nowarn plot(sol)
                @test_nowarn plot(sol, denseplot=false)
                message = "This plot does not visualize any uncertainties"
                @test_logs (:warn, message) plot(sol, idxs=(1, 2))
                @test_logs (:warn, message) plot(sol, idxs=(1, 1, 2))
                @test_nowarn plot(sol, tspan=prob.tspan)
            end

            @testset "Mean Solution" begin
                msol = mean(sol)
                x = @test_nowarn msol(prob.tspan[1])
                @test x isa AbstractArray
                xs = @test_nowarn msol(sol.t[1:2])
                @test xs isa ProbNumDiffEq.DiffEqArray
                @test xs.u isa AbstractArray{<:AbstractArray}
                @test_nowarn msol
                @test_nowarn plot(msol)
            end
        end
    end
end
