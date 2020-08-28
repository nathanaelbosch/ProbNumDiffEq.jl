# Goal: Make sure that our solvers are "correct" for small steps or tolerances
# Verify this for many (ideally all) combinations
# Compare with an algorithm from OrdinaryDiffEq.jl with high precision
using Test
using OrdinaryDiffEq
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo
using ModelingToolkit

import ProbNumODE: remake_prob_with_jac


@testset "Constant steps" begin
    prob = prob_ode_lotkavoltera
    prob = remake_prob_with_jac(prob)

    true_sol = solve(prob, Tsit5(), abstol=1e-15, reltol=1e-15)

    for method in (EKF0(), EKF1()),
        sigma in (:fixedMLE, :schober),
        error in (:schober, :prediction, :filtering),
        q in 1:3

        sol = solve(prob, method, q=q,
                    steprule=:constant, dt=1e-4,
                    sigmarule=sigma,
                    local_errors=error,
                    smooth=false,
                    )
        diffs = sol.u .- true_sol.(sol.t)
        maxdiff = maximum(abs.(ProbNumODE.stack(diffs)))
        mse = sum(norm.(diffs, 2)) / length(sol.t)

        @test mse < 1e-7
        @test maxdiff < 1e-7
    end
end



@testset "Standard adaptive steps" begin
    prob = prob_ode_lotkavoltera
    prob = remake_prob_with_jac(prob)

    t_eval = prob.tspan[1]:0.01:prob.tspan[end]
    sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
    true_sol = sol.(t_eval)

    for method in (EKF0(), EKF1()),
        # sigma in (:fixedMLE, :schober),
        sigma in [:schober],
        error in (:schober, :prediction, :filtering),
        q in 1:3

        sol = solve(prob, method, q=q,
                    steprule=:standard, abstol=1e-9, reltol=1e-9,
                    sigmarule=sigma,
                    local_errors=error,
                    smooth=false,
                    )
        diffs = sol.(t_eval) .- true_sol
        maxdiff = maximum(abs.(ProbNumODE.stack(diffs)))
        mse = sum(norm.(diffs, 2)) / length(t_eval)

        @test mse < 1e-5
        @test maxdiff < 1e-5
    end
end
