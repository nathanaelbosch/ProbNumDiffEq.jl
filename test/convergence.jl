using Test
using ProbNumDiffEq
using DiffEqDevTools


TESTTOL = 0.2


# Simple linear problem
prob = ODEProblem(
    ODEFunction(
        (u, p, t) -> 1.01 .* u,
        jac=(u, p, t) -> 1.01;
        analytic=(u0, p, t) -> u0 .* exp(1.01*t),
    ), [big(1/2)], big.((0.0, 1.0)))


@testset "EK0(order=$q) convergence" for q in 1:5
    dts = 1 .//2 .^(8:-1:4)
    sim = test_convergence(dts, prob, EK0(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=0.2
    @test sim.𝒪est[:l2] ≈ q+1 atol=0.25
    @test sim.𝒪est[:l∞] ≈ q+1 atol=0.2
end



@testset "EK1(order=$q) convergence" for q in 1:5
    dts = 1 .//2 .^(8:-1:4)
    sim = test_convergence(dts, prob, EK1(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=0.55
    @test sim.𝒪est[:l2] ≈ q+1 atol=0.5
    @test sim.𝒪est[:l∞] ≈ q+1 atol=0.5
end
