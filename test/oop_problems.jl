using ProbNumDiffEq
using ModelingToolkit
using OrdinaryDiffEq
using LinearAlgebra
using Test

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

@testset "OOP problem" begin
    f(u, p, t) = p .* u .* (1 .- u)
    prob = ODEProblem(f, [1e-1], (0.0, 2.0), [3.0])
    @testset "without jacobian" begin
        # first without defined jac
        @test solve(prob, EK0(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob, EK1(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob, EK1(order=4, initialization=ClassicSolverInit(Tsit5()))) isa
              ProbNumDiffEq.ProbODESolution
    end
    @testset "with jacobian" begin
        # now with defined jac using the new MTK API with symbolic maps
        prob_mtk = make_mtk_problem(prob)
        @test solve(prob_mtk, EK0(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob_mtk, EK1(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob_mtk, EK1(order=4, initialization=ClassicSolverInit(Tsit5()))) isa
              ProbNumDiffEq.ProbODESolution
    end
    @testset "with mass matrix" begin
        function rober(u, p, t)
            y₁, y₂, y₃ = u
            k₁, k₂, k₃ = p
            return [
                -k₁ * y₁ + k₃ * y₂ * y₃,
                k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2,
                y₁ + y₂ + y₃ - 1,
            ]
        end
        f = ODEFunction(rober, mass_matrix=Diagonal([1, 1, 0]))
        prob = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4))
        sol = solve(prob, EK1(order=3))
        @test sol.retcode == ReturnCode.Success
        @test all(u -> abs(sum(u) - 1) < 1e-8, sol.u)
        @test_throws "DAE" solve(prob, EK0())
    end
end
