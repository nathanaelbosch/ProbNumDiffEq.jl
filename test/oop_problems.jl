using ProbNumDiffEq
using Test

@testset "OOP problem" begin
    f(u, p, t) = p .* u .* (1 .- u)
    prob = ODEProblem(f, [1e-1], (0.0, 2.0), [3.0])
    @testset "without jacobian" begin
        # first without defined jac
        @test_nowarn solve(prob, EK0(order=4))
        @test_nowarn solve(prob, EK1(order=4))
        @test_nowarn solve(prob, EK1(order=4, initialization=ClassicSolverInit()))
    end
    @testset "with jacobian" begin
        # now with defined jac
        prob = ODEProblem(modelingtoolkitize(prob), prob.u0, prob.tspan, jac=true)
        @test_nowarn solve(prob, EK0(order=4))
        @test_nowarn solve(prob, EK1(order=4))
        @test_nowarn solve(prob, EK1(order=4, initialization=ClassicSolverInit()))
    end
end
