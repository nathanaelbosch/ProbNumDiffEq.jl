using Test
using ProbNumDiffEq
import ProbNumDiffEq as PNDE

@testset "Standard ODEProblem" begin
    # Use ROBER so that there is a mass matrix
    function rober(u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        return [
            -k₁ * y₁ + k₃ * y₂ * y₃,
            k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2,
            y₁ + y₂ + y₃ - 1,
        ]
    end
    function rober!(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
        du[3] = y₁ + y₂ + y₃ - 1
        return nothing
    end
    M = [1.0 0 0; 0 1.0 0; 0 0 0]

    u0, tspan, p = [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4)

    d, q = length(u0), 5
    x, z = rand(d * (q + 1)), rand(d)
    t = tspan[1]

    E0 = PNDE.projection(d, q)(0)
    E1 = PNDE.projection(d, q)(1)

    @testset "IIP" begin
        f = ODEFunction(rober!, mass_matrix=M)
        prob = ODEProblem(f, u0, tspan, p)

        m = PNDE.make_measurement_model(prob.f)
        @test_nowarn m(z, x, p, t)

        du = copy(z)
        f(du, E0 * x, p, t)
        @test z ≈ f.mass_matrix * E1 * x - du
    end

    @testset "OOP" begin
        f = ODEFunction(rober, mass_matrix=M)
        prob = ODEProblem(f, u0, tspan, p)

        m = PNDE.make_measurement_model(prob.f)
        @test_broken m(x, p, t)
        @test_broken m(x, p, t) ≈ f.mass_matrix * E1 * x - f(E0 * x, p, t)
    end
end

@testset "SecondOrderODEProblem" begin
    f(du, u, p, t) = p .* u
    f!(ddu, du, u, p, t) = ddu .= p .* u
    du0, u0, tspan, p = [0.0], [1.0], (0.0, 1.0), [0.9]

    d, q = length(u0), 5
    x, z = rand(d * (q + 1)), rand(d)
    t = tspan[1]

    E0 = PNDE.projection(d, q)(0)
    E1 = PNDE.projection(d, q)(1)
    E2 = PNDE.projection(d, q)(2)

    @testset "IIP" begin
        prob = SecondOrderODEProblem(f!, du0, u0, tspan, p)

        m = PNDE.make_measurement_model(prob.f)
        @test_nowarn m(z, x, p, t)

        ddu = copy(z)
        prob.f.f1(ddu, E1 * x, E0 * x, p, t)
        @test z ≈ E2 * x - ddu
    end

    @testset "OOP" begin
        prob = SecondOrderODEProblem(f, du0, u0, tspan, p)

        m = PNDE.make_measurement_model(prob.f)
        @test_broken m(x, p, t)
        @test_broken m(x, p, t) ≈ E2 * x - prob.f.f1(E1 * x, E0 * x, p, t)
    end
end
