# Goal: Make sure some combinations that raise errors do so
using Test
using OrdinaryDiffEq
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

import ProbNumDiffEq: remake_prob_with_jac


@testset "One-dim problems don't work so far!" begin
    prob = prob_ode_linear
    @test_throws ErrorException solve(prob, EK0())
end

@testset "Fixed-timestep requires dt" begin
    prob = prob_ode_lotkavoltera
    @test_throws ErrorException solve(prob, EK0(), adaptive=false)
    @test solve(prob, EK0(), adaptive=false, dt=0.05) isa ProbNumDiffEq.ProbODESolution
end
