using Test
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo



@testset "Correctness for different sigmas" begin
    prob = prob_ode_fitzhughnagumo
    true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)

    @testset "Schober Sigma" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:schober)
        @test sol.u ≈ true_sol.(sol.t)
    end

    @testset "Fixed MLE" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:fixedMLE)
        @test sol.u ≈ true_sol.(sol.t)
    end

    @testset "Fixed MAP" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:fixedMAP)
        @test sol.u ≈ true_sol.(sol.t)
    end

    @testset "Fixed weighted MLE" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:fixedWeightedMLE)
        @test sol.u ≈ true_sol.(sol.t)
    end

    @testset "Dynamic one-step EM" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:EM)
        @test sol.u ≈ true_sol.(sol.t)
    end

    # @testset "Optim-based" begin
    #     sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:optim)
    #     @test sol.u ≈ true_sol.(sol.t)
    # end
end
