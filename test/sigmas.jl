using Test
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo
using ModelingToolkit



@testset "Correctness for different sigmas" begin
    prob = prob_ode_fitzhughnagumo

    @testset "Schober Sigma" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:schober)
        true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
        @test sol.u.μ ≈ true_sol.(sol.t)
    end

    @testset "Fixed MLE" begin
        sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:fixedMLE)
        true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
        @test sol.u.μ ≈ true_sol.(sol.t)
    end

    @testset "Fixed MAP" begin
        @test_broken begin
            sol = solve(prob, EKF0(), steprule=:constant, dt=1e-4, sigmarule=:fixedMAP)
            true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
            sol.u.μ ≈ true_sol.(sol.t)
        end
    end
end
