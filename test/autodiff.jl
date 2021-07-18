using ProbNumDiffEq
using Test
using LinearAlgebra
using ForwardDiff, Zygote, ReverseDiff

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo


prob = ProbNumDiffEq.remake_prob_with_jac(prob_ode_fitzhughnagumo)

function param_to_loss(p)
    sol = solve(remake(prob, p=p), EK1(order=3))
    return norm(sol.u[end])  # Dummy loss
end
function startval_to_loss(u0)
    sol = solve(remake(prob, u0=u0), EK1(order=3))
    return norm(sol.u[end])  # Dummy loss
end

dldp = [0.026680212891877435, -0.028019989130281753, 0.3169977494388167, 0.6749351039218744]
dldu0 = [0.6500925873857853, -0.004812245513746423]

@testset "ForwardDiff.jl" begin
    @test ForwardDiff.gradient(param_to_loss, prob.p) ≈ dldp
    @test ForwardDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0
end

@testset "ReverseDiff.jl" begin
    @test_broken ReverseDiff.gradient(param_to_loss, prob.p) ≈ dldp
    @test_broken ReverseDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0
end

@testset "Zygote.jl" begin
    @test_broken Zygote.gradient(param_to_loss, prob.p) ≈ dldp
    @test_broken Zygote.gradient(startval_to_loss, prob.u0) ≈ dldu0
end
