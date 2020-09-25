# Goal: Make sure that our solvers are "correct" for small steps or tolerances
# Verify this for many (ideally all) combinations
# Compare with an algorithm from OrdinaryDiffEq.jl with high precision
using ProbNumODE
using Test
using OrdinaryDiffEq
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

import ProbNumODE: remake_prob_with_jac


for (prob, probname) in [
    (remake_prob_with_jac(prob_ode_lotkavoltera), "lotkavoltera"),
    (remake_prob_with_jac(prob_ode_fitzhughnagumo), "fitzhughnagumo"),
]
    @testset "Constant steps: $probname" begin

        true_sol = solve(remake(prob, u0=big.(prob.u0)), Tsit5(), abstol=1e-20, reltol=1e-20)

        for method in (EKF0(), EKF1()),
            sigma in [:fixedMLE, :schober, :EM, :fixedWeightedMLE, :fixedMAP],
            q in 1:4
            @testset "Constant steps: $probname; q=$q, sigma=$sigma, methdo=$method" begin

            @debug "Testing for correctness: Constant steps" probname method sigma q dt

            sol = solve(prob, method, q=q,
                        steprule=:constant, dt=1e-4,
                        sigmarule=sigma,
                        smooth=false,
                        )
            @test sol.u ≈ true_sol.(sol.t) rtol=1e-6
            end
        end
    end
end



for (prob, probname) in [
    (remake_prob_with_jac(prob_ode_lotkavoltera), "lotkavoltera"),
    (remake_prob_with_jac(prob_ode_fitzhughnagumo), "fitzhughnagumo"),
]
    @testset "Adaptive steps: $probname" begin

        t_eval = prob.tspan[1]:0.01:prob.tspan[end]
        true_sol = solve(remake(prob, u0=big.(prob.u0)), Tsit5(), abstol=1e-20, reltol=1e-20)
        true_dense_vals = true_sol.(t_eval)

        for method in (EKF0(), EKF1()),
            sigma in [:schober, :fixedMLE, :EM, :fixedWeightedMLE, :fixedMAP],
            q in 1:4
            @testset "Adaptive steps: $probname; q=$q, sigma=$sigma, methdo=$method" begin

            @debug "Testing for correctness: Adaptive steps" probname method sigma q

            sol = solve(prob, method, q=q,
                        steprule=:standard, abstol=1e-9, reltol=1e-9,
                        sigmarule=sigma,
                        smooth=false,
                        )

            @test sol.u ≈ true_sol.(sol.t) rtol=1e-6
            @test sol.(t_eval) ≈ true_dense_vals rtol=1e-6

            end
        end
    end
end
