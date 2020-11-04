# Everytime I encounter something that raises some error and I fix it, I should add that
# specific problem to this list to make sure, that this specific run then works without
# bugs.
using ODEFilters
using Test
using LinearAlgebra


using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_vanstiff


@testset "Smoothing with small constant steps" begin
    prob = ODEFilters.remake_prob_with_jac(prob_ode_fitzhughnagumo)
    @test solve(prob, EKF0(order=4, diffusionmodel=:fixed, smooth=true),
                adaptive=false, dt=1e-3) isa ODEFilters.ProbODESolution
    @test solve(prob, EKF1(order=4, diffusionmodel=:fixed, smooth=true),
                adaptive=false, dt=1e-3) isa ODEFilters.ProbODESolution
end


@testset "Stiff Vanderpol" begin
    prob = ODEFilters.remake_prob_with_jac(prob_ode_vanstiff)
    @test solve(prob, EKF1(order=3)) isa ODEFilters.ProbODESolution
end


@testset "Big Float" begin
    prob = remake(prob_ode_fitzhughnagumo, u0=big.(u0))
    @test_broken solve(prob, EKF0(order=3)) isa ODEFilters.ProbODESolution
end


@testset "OOP problem definition" begin
    prob = ODEProblem((u, p, t) -> ([p[1] * u[1] .* (1 .- u[1])]), [1e-1], (0.0, 5), [3.0])
    @test solve(prob, EKF0(order=4)) isa ODEFilters.ProbODESolution
    prob = ODEFilters.remake_prob_with_jac(prob)
    @test solve(prob, EKF1(order=4)) isa ODEFilters.ProbODESolution
end
