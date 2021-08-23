using Test
using ProbNumDiffEq
using DiffEqDevTools


# Simple linear problem
prob = ODEProblem(
    ODEFunction(
        (u, p, t) -> 1.01 .* u,
        jac=(u, p, t) -> 1.01;
        analytic=(u0, p, t) -> u0 .* exp(1.01*t),
    ), [big(1/2)], big.((0.0, 1.0)))

# Step-sizes
dts1 = 1 .//2 .^(7:-1:4)
dts2 = 1 .//2 .^(10:-1:8)


@testset "EK0(order=$q) convergence" for q in 1:5
    sim = test_convergence(dts1, prob, EK0(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=0.3
    @test sim.𝒪est[:l2] ≈ q+1 atol=0.3
    @test sim.𝒪est[:l∞] ≈ q+1 atol=0.3
end
@testset "EK0(order=$q) convergence" for q in [7, 8]
    sim = test_convergence(dts2, prob, EK0(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=1.5
    @test sim.𝒪est[:l2] ≈ q+1 atol=1
    @test sim.𝒪est[:l∞] ≈ q+1 atol=1
end



@testset "EK1(order=$q) convergence" for q in 1:5
    sim = test_convergence(dts1, prob, EK1(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=0.55
    @test sim.𝒪est[:l2] ≈ q+1 atol=0.5
    @test sim.𝒪est[:l∞] ≈ q+1 atol=0.5
end
@testset "EK1(order=$q) convergence" for q in [7, 8, 10]
    sim = test_convergence(dts2, prob, EK1(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=1
    @test sim.𝒪est[:l2] ≈ q+1 atol=1
    @test sim.𝒪est[:l∞] ≈ q+1 atol=1
end
