using ProbNumDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Test

function twobody(du, u, p, t)
    R3 = norm(u[1:2])^3
    @. du[3:4] = -u[1:2] / R3
    @. du[1:2] = u[3:4]
end
u0, du0 = [0.4, 0.0], [0.0, 2.0]
tspan = (0, 0.1)
prob_base = ODEProblem(twobody, [u0...; du0...], tspan)

function twobody2_iip(ddu, du, u, p, t)
    R3 = norm(u)^3
    @. ddu = -u / R3
end
prob_iip = SecondOrderODEProblem(twobody2_iip, du0, u0, tspan)

function twobody2_oop(du, u, p, t)
    R3 = norm(u)^3
    return -u / R3
end
prob_oop = SecondOrderODEProblem(twobody2_oop, du0, u0, tspan)

appxsol = solve(prob_iip, Vern9(), abstol=1e-10, reltol=1e-10)

@testset "$S" for (S, _prob) in (("IIP", prob_iip), ("OOP", prob_oop))
    @testset "$alg" for alg in (
        EK0(),
        EK1(),
        EK0(initialization=ForwardDiffInit(2)),
        EK1(initialization=ForwardDiffInit(2)),
        # EK1(initialization=ClassicSolverInit()), # unstable for this problem
        EK1(diffusionmodel=FixedDiffusion()),
        EK0(diffusionmodel=FixedMVDiffusion()),
        EK0(diffusionmodel=DynamicMVDiffusion()),
    )
        sol = solve(_prob, alg)

        @test sol isa ProbNumDiffEq.ProbODESolution
        @test sol.u[end] ≈ appxsol.u[end] rtol = 1e-3

        @test sol(tspan[2] / π).μ ≈ appxsol(tspan[2] / π) rtol = 1e-3
    end
end
@testset "ClassicSolverInit" begin
    @test_nowarn solve(prob_iip, EK1(initialization=ClassicSolverInit()))
end
