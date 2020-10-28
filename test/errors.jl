# Goal: Make sure some combinations that raise errors do so
using Test
using OrdinaryDiffEq
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo
using ModelingToolkit

import ODEFilters: remake_prob_with_jac


@testset "EKF1 requires Jac" begin
    prob = prob_ode_lotkavoltera
    @test_throws ErrorException solve(prob, EKF1())
    @test solve(remake_prob_with_jac(prob), EKF1()) isa ODEFilters.ProbODESolution
end

@testset "One-dim problems don't work so far!" begin
    prob = prob_ode_linear
    @test_throws ErrorException solve(prob, EKF0())
end

@testset "Fixed-timestep requires dt" begin
    prob = prob_ode_lotkavoltera
    @test_throws ErrorException solve(prob, EKF0(), adaptive=false)
    @test solve(prob, EKF0(), adaptive=false, dt=0.05) isa ODEFilters.ProbODESolution
end
