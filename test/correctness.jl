# Make sure that our solvers with small steps return correct results
# Compare with DifferentialEquations.jl.
using Test
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo
using ModelingToolkit



function test_prob_solution_correctness(prob, atol=1e-6, args...; kwargs...)

    prob = remake(prob, p=collect(prob.p))

    sol = solve(prob, args...; kwargs...)
    true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)

    our_u = sol.u.μ
    true_u = true_sol.(sol.t)

    @test our_u[end] ≈ true_u[end] atol=atol
    @test our_u ≈ true_u atol=atol
end


@testset "EKF0 with constant steps" begin
    @test_broken test_prob_solution_correctness(prob_ode_linear, EKF0(), steprule=:constant, dt=1e-4)
    @test_broken test_prob_solution_correctness(prob_ode_2Dlinear, EKF0(), steprule=:constant, dt=1e-4)
    test_prob_solution_correctness(
        prob_ode_lotkavoltera, EKF0(), steprule=:constant, dt=1e-4)
    test_prob_solution_correctness(
        prob_ode_fitzhughnagumo, EKF0(), steprule=:constant, dt=1e-4)
end


function remake_prob_with_jac(prob)
    prob = remake(prob, p=collect(prob.p))
    sys = modelingtoolkitize(prob)
    jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
    f = ODEFunction(prob.f.f, jac=jac)
    return remake(prob, f=f)
end

@testset "EKF1 with constant steps" begin
    test_prob_solution_correctness(
        remake_prob_with_jac(prob_ode_lotkavoltera),
        EKF1(), steprule=:constant, dt=1e-4)
    test_prob_solution_correctness(
        remake_prob_with_jac(prob_ode_fitzhughnagumo),
        EKF1(), steprule=:constant, dt=1e-3)
end
