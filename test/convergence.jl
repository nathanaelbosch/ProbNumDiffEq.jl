using Test
using ProbNumDiffEq
using DiffEqDevTools

# Simple linear problem
linear(du, u, p, t) = du .= p .* u
linear_jac(J, u, p, t) = J .= p
linear_analytic(u0, p, t) = @. u0 * exp(p * t)
const prob = ODEProblem(
    ODEFunction(linear, jac=linear_jac, analytic=linear_analytic),
    [big(1 / 2)],
    big.((0.0, 1.0)),
    big(1.01),
)

# Step-sizes
const dts1 = (1 / 2) .^ (7:-1:4)
const dts2 = (1 / 2) .^ (8:-1:6)
const dts3 = (1 / 2) .^ (12:-1:8)
@testset "EK0 convergence" begin
    @testset "order=$q" for q in (1, 2, 3, 4)
        sim = test_convergence(dts1, prob, EK0(order=q))
        @test sim.𝒪est[:final] ≈ q + 1 atol = 0.3
        @test sim.𝒪est[:l2] ≈ q + 1 atol = 0.3
        @test sim.𝒪est[:l∞] ≈ q + 1 atol = 0.3
    end
    @testset "order=$q" for q in (5, 6)
        sim = test_convergence(dts2, prob, EK0(order=q))
        @test sim.𝒪est[:final] ≈ q + 1 atol = 0.3
        @test sim.𝒪est[:l2] ≈ q + 1 atol = 0.3
        @test sim.𝒪est[:l∞] ≈ q + 1 atol = 0.3
    end
    @testset "order=$q" for q in (7, 8)
        sim = test_convergence(dts3, prob, EK0(order=q))
        @test sim.𝒪est[:final] ≈ q + 1 atol = 1.5
        @test sim.𝒪est[:l2] ≈ q + 1 atol = 1
        @test sim.𝒪est[:l∞] ≈ q + 1 atol = 0.3
    end
end

@testset "EK1 convergence" begin
    @testset "order=$q" for q in (1, 2, 3, 4, 5, 6)
        sim = test_convergence(dts2, prob, EK0(order=q))
        @test sim.𝒪est[:final] ≈ q + 1 atol = 0.3
        @test sim.𝒪est[:l2] ≈ q + 1 atol = 0.3
        @test sim.𝒪est[:l∞] ≈ q + 1 atol = 0.3
    end
    @testset "order=$q" for q in (7, 8)
        sim = test_convergence(dts3, prob, EK0(order=q))
        @test sim.𝒪est[:final] ≈ q + 1 atol = 1.5
        @test sim.𝒪est[:l2] ≈ q + 1 atol = 1
        @test sim.𝒪est[:l∞] ≈ q + 1 atol = 0.3
    end
end
