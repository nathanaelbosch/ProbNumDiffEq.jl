using ProbNumDiffEq
using ModelingToolkit
using Test
using LinearAlgebra
using FiniteDifferences
using ForwardDiff
import SciMLStructures
# using ReverseDiff
# using Zygote

import ODEProblemLibrary: prob_ode_fitzhughnagumo

# Helper function to convert a regular ODE problem to MTK form with symbolic maps
function make_mtk_problem(_prob)
    sys = structural_simplify(modelingtoolkitize(_prob))
    unknowns_list = ModelingToolkit.unknowns(sys)
    params_list = ModelingToolkit.parameters(sys)
    u0_map = Dict(unknowns_list[i] => _prob.u0[i] for i in eachindex(_prob.u0))
    p_vals = collect(_prob.p)
    p_map = Dict(params_list[i] => p_vals[i] for i in eachindex(p_vals))
    return ODEProblem(sys, merge(u0_map, p_map), _prob.tspan; jac=true)
end

@testset "solver: $ALG" for ALG in (EK0, EK1, DiagonalEK1)
    _prob = prob_ode_fitzhughnagumo
    prob = make_mtk_problem(_prob)
    ps = ModelingToolkit.parameter_values(prob)
    ps = SciMLStructures.replace(SciMLStructures.Tunable(), ps, [1.0, 2.0, 3.0, 4.0])
    prob = remake(prob, p=ps)

    function param_to_loss(p)
        ps = ModelingToolkit.parameter_values(prob)
        ps = SciMLStructures.replace(SciMLStructures.Tunable(), ps, p)
        sol = solve(
            remake(prob, p=ps),
            ALG(order=3, smooth=false),
            sensealg=SensitivityADPassThrough(),
            abstol=1e-6,
            reltol=1e-5,
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
            abstol=1e-6,
            reltol=1e-5,
            save_everystep=false,
            dense=false,
        )
        return norm(sol.u[end])  # Dummy loss
    end

    p, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)

    #dldp = FiniteDiff.finite_difference_gradient(param_to_loss, p)
    #dldu0 = FiniteDiff.finite_difference_gradient(startval_to_loss, prob.u0)
    # For some reason FiniteDiff.jl is not working anymore so we use FiniteDifferences.jl:
    dldp = grad(central_fdm(5, 1), param_to_loss, p)[1]
    dldu0 = grad(central_fdm(5, 1), startval_to_loss, prob.u0)[1]

    @testset "ForwardDiff.jl" begin
        @test ForwardDiff.gradient(param_to_loss, p) ≈ dldp rtol = 1e-2
        @test ForwardDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0 rtol = 5e-2
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
