using ProbNumDiffEq
using ModelingToolkit
using Test
using LinearAlgebra
using FiniteDiff
using ForwardDiff
# using ReverseDiff
# using Zygote

import ODEProblemLibrary: prob_ode_fitzhughnagumo

@testset "solver: $ALG" for ALG in (EK0, EK1, DiagonalEK1)

    _prob = prob_ode_fitzhughnagumo
    prob = ODEProblem(modelingtoolkitize(_prob), _prob.u0, _prob.tspan, jac=true)
    function param_to_loss(p)
        sol = solve(
            remake(prob, p=p),
            ALG(order=3, smooth=false),
            sensealg=SensitivityADPassThrough(),
            abstol=1e-3,
            reltol=1e-2,
            save_everystep=false,
            dense=false,
        )
        return norm(sol.u[end])  # Dummy loss
    end
    function startval_to_loss(u0)
        sol = solve(
            remake(prob, u0=u0),
            ALG(order=3, smooth=false),
            sensealg=SensitivityADPassThrough(),
            abstol=1e-3,
            reltol=1e-2,
            save_everystep=false,
            dense=false,
        )
        return norm(sol.u[end])  # Dummy loss
    end

    dldp = FiniteDiff.finite_difference_gradient(param_to_loss, prob.p)
    dldu0 = FiniteDiff.finite_difference_gradient(startval_to_loss, prob.u0)

    @testset "ForwardDiff.jl" begin
        @test ForwardDiff.gradient(param_to_loss, prob.p) ≈ dldp rtol = 1e-3
        @test ForwardDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0 rtol = 5e-3
    end

    # @testset "ReverseDiff.jl" begin
    #     @test_broken ReverseDiff.gradient(param_to_loss, prob.p) ≈ dldp
    #     @test_broken ReverseDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0
    # end

    # @testset "Zygote.jl" begin
    #     @test_broken Zygote.gradient(param_to_loss, prob.p) ≈ dldp
    #     @test_broken Zygote.gradient(startval_to_loss, prob.u0) ≈ dldu0
    # end
end
