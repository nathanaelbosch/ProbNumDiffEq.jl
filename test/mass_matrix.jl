using ProbNumDiffEq
using OrdinaryDiffEq
using Test

@testset "Robertson in mass-matrix-ODE form" begin
    function rober(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
        du[3] = y₁ + y₂ + y₃ - 1
        return nothing
    end
    M = [
        1.0 0 0
        0 1.0 0
        0 0 0
    ]
    f = ODEFunction(rober, mass_matrix=M)
    prob = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4))

    sol1 = solve(prob, EK1(order=3))
    sol2 = solve(prob, RadauIIA5())
    @test sol1[end] ≈ sol2[end] rtol = 1e-5
end
