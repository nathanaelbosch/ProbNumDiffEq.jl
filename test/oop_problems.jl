using ProbNumDiffEq
using ModelingToolkit
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
        @test solve(prob, EK1(order=4, initialization=ClassicSolverInit())) isa
              ProbNumDiffEq.ProbODESolution
    end
    @testset "with jacobian" begin
        # now with defined jac using the new MTK API with symbolic maps
        prob_mtk = make_mtk_problem(prob)
        @test solve(prob_mtk, EK0(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob_mtk, EK1(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob_mtk, EK1(order=4, initialization=ClassicSolverInit())) isa
              ProbNumDiffEq.ProbODESolution
    end
end
