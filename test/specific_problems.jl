# Everytime I encounter something that raises some error and I fix it, I should add that
# specific problem to this list to make sure, that this specific run then works without
# bugs.
using ProbNumODE
using Test
using LinearAlgebra


using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_vanstiff


@testset "Smoothing with small constant steps" begin
    prob = ProbNumODE.remake_prob_with_jac(prob_ode_fitzhughnagumo)
    @test solve(prob, EKF0(order=4, diffusionmodel=:fixed, smooth=true),
                adaptive=false, dt=1e-3) isa ProbNumODE.ProbODESolution
    @test solve(prob, EKF1(order=4, diffusionmodel=:fixed, smooth=true),
                adaptive=false, dt=1e-3) isa ProbNumODE.ProbODESolution
end


@testset "Stiff Vanderpol" begin
    prob = ProbNumODE.remake_prob_with_jac(prob_ode_vanstiff)
    @test_broken solve(prob, EKF1(order=3)) isa ProbNumODE.ProbODESolution
end
