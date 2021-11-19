using ProbNumDiffEq
using Test
using LinearAlgebra

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_vanstiff

@testset "Smoothing with small constant steps" begin
    prob = ProbNumDiffEq.remake_prob_with_jac(prob_ode_fitzhughnagumo)
    @test solve_ieks(prob, IEKS(order=4, diffusionmodel=FixedDiffusion())) isa
          ProbNumDiffEq.ProbODESolution
end
