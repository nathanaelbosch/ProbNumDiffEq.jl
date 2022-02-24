using ProbNumDiffEq
using ModelingToolkit
using Test
using LinearAlgebra
using FiniteDiff, ForwardDiff, Zygote, ReverseDiff

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo

prob = prob_ode_fitzhughnagumo
prob = ODEProblem(modelingtoolkitize(prob), prob.u0, prob.tspan, jac=true)

function param_to_loss(p)
    sol = solve(remake(prob, p=p), EK1(order=3), sensealg=SensitivityADPassThrough())
    return norm(sol.u[end])  # Dummy loss
end
function startval_to_loss(u0)
    sol = solve(remake(prob, u0=u0), EK1(order=3), sensealg=SensitivityADPassThrough())
    return norm(sol.u[end])  # Dummy loss
end

dldp = FiniteDiff.finite_difference_gradient(param_to_loss, prob.p)
dldu0 = FiniteDiff.finite_difference_gradient(startval_to_loss, prob.u0)

@testset "ForwardDiff.jl" begin
    @test ForwardDiff.gradient(param_to_loss, prob.p) ≈ dldp rtol = 1e-4
    @test ForwardDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0 rtol = 5e-4
end

@testset "ReverseDiff.jl" begin
    @test_broken ReverseDiff.gradient(param_to_loss, prob.p) ≈ dldp
    @test_broken ReverseDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0
end

@testset "Zygote.jl" begin
    @test_broken Zygote.gradient(param_to_loss, prob.p) ≈ dldp
    @test_broken Zygote.gradient(startval_to_loss, prob.u0) ≈ dldu0
end
