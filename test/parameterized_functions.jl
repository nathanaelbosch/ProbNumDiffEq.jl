using ProbNumDiffEq
using Test
using ParameterizedFunctions

@testset "Problem definition with ParameterizedFunctions.jl" begin
    f = @ode_def LotkaVolterra begin
        dx = a * x - b * x * y
        dy = -c * y + d * x * y
    end a b c d
    p = [1.5, 1, 3, 1]
    tspan = (0.0, 1.0)
    u0 = [1.0, 1.0]
    prob = ODEProblem(f, u0, tspan, p)
    @test solve(prob, EK1(order=3)) isa ProbNumDiffEq.ProbODESolution
end
