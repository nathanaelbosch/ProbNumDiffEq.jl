using Test
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo
using ModelingToolkit



@testset "Correctness for different sigmas" begin
    test_prob_solution_correctness(
        prob_ode_fitzhughnagumo, EKF0(), steprule=:constant, dt=1e-4,
        sigmarule=:schober)
    test_prob_solution_correctness(
        prob_ode_fitzhughnagumo, EKF0(), steprule=:constant, dt=1e-4,
        sigmarule=:fixedMLE)
    test_prob_solution_correctness(
        prob_ode_fitzhughnagumo, EKF0(), steprule=:constant, dt=1e-4,
        sigmarule=:fixedMAP)
end
