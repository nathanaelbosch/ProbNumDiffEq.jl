using ProbNumODE
using Test
using Measurements
using Plots


@testset "ProbNumODE" begin
    @testset "Priors" begin include("priors.jl") end
end


@testset "Fitzhugh-Nagumo Correctness" begin
    prob = fitzhugh_nagumo()
    sol = solve(prob, EKF0(), steprule=:constant, dt=0.001, q=1)
    @test length(sol) > 2
    @test length(sol.t) == length(sol.u)
    @test length(prob.u0) == length(sol.u[end])

    true_sol = [2.010422386552278; 0.6382569402421574]  # Computed with DifferentialEquations.jl
    @test Measurements.values.(sol[end]) ≈ true_sol atol=0.01
end

@testset "Methods" begin
    prob = fitzhugh_nagumo()
    sol = solve(prob, EKF0(), smooth=false)
    sol = solve(prob, EKF1(), smooth=false)
end

@testset "Plotting" begin
    prob = fitzhugh_nagumo()
    sol = solve(prob, EKF0(), smooth=false)
    plot(sol)
end

@testset "Sigmas" begin
    prob = fitzhugh_nagumo()
    # Multiple different methods
    sol = solve(prob, EKF0(), steprule=:standard, sigmarule=:schober,
                abstol=1e-3, reltol=1e-3, q=2)
    @test_broken begin
        sol = solve(prob, EKF0(), steprule=:constant, sigmarule=:fixedMLE,
                    abstol=1e-1, reltol=1e-1, q=2)
    end
    # sol = solve(prob, EKF0(), steprule=:constant, sigmarule=:fixedMAP,
    #             abstol=1e-1, reltol=1e-1, q=2)
    # sol = solve(prob, EKF0(), steprule=:constant, sigmarule=ProbNumODE.WeightedMLESigma(),
    #             abstol=1e-1, reltol=1e-1, q=2)
end

@testset "Error Estimations" begin
    prob = fitzhugh_nagumo()
    # Multiple different methods
    sol = solve(prob, EKF0(), steprule=:standard, local_errors=:schober,
                abstol=1e-3, reltol=1e-3, q=2)
    # sol = solve(prob, EKF0(), steprule=:standard, local_errors=:prediction,
    #             abstol=1e-3, reltol=1e-3, q=2)
    # sol = solve(prob, EKF0(), steprule=:standard, local_errors=:filtering,
    #             abstol=1e-3, reltol=1e-3, q=2)
end
