using ProbNumDiffEq
using DiffEqDevTools


TESTTOL = 0.2


# Simple linear problem
prob = ODEProblem(
    ODEFunction(
        (u, p, t) -> [1.01].*u,
        jac=(u, p, t) -> [1.01];
        analytic=(u0, p, t) -> u0.*[exp(1.01*t)],
    ), [big(1/2)], (0.0, 1.0))


@testset "EK0(order=$q)" for q in 1:3
    dts = 1 .//2 .^(9:-1:2)
    sim = test_convergence(dts, prob, EK0(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=TESTTOL
    @test sim.𝒪est[:l2] ≈ q+1 atol=TESTTOL
    @test sim.𝒪est[:l∞] ≈ q+1 atol=TESTTOL
end
@testset "EK0(order=$q)" for q in 4:5
    dts = 1 .//2 .^(8:-1:4)
    sim = test_convergence(dts, prob, EK0(order=q))
    @test sim.𝒪est[:final] ≈ q+1 atol=TESTTOL
    @test sim.𝒪est[:l2] ≈ q+1 atol=TESTTOL+0.1
    @test sim.𝒪est[:l∞] ≈ q+1 atol=TESTTOL+0.1
end



@testset "EK1(order=$q)" for q in [1, 3, 4, 5]
    dts = 1 .//2 .^(8:-1:3)
    sim = test_convergence(dts, prob, EK1(order=q))
    @test sim.𝒪est[:l2] ≈ q+1 atol=TESTTOL+0.1
end
