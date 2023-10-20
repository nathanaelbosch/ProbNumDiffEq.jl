using ProbNumDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Test

@testset "Simple UniformScaling mass-matrix" begin
    vf(du, u, p, t) = (du .= u)
    M = -100I
    f = ODEFunction(vf, mass_matrix=M)
    prob = ODEProblem(f, [1.0], (0.0, 1.0))
    ref = solve(prob, RadauIIA5())
    sol = solve(prob, EK1(order=3))
    @test sol[end] ≈ ref[end] rtol = 1e-10
end

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
        1 0 0
        0 1 0
        0 0 0
    ]
    f = ODEFunction(rober, mass_matrix=M)
    prob = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4))

    ref = solve(prob, EK1(order=3))
    sol = solve(prob, RadauIIA5())
    @test sol[end] ≈ ref[end] rtol = 1e-8
end
