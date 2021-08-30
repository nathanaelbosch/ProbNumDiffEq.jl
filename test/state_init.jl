# Goal: Test the correctness of state initialization
using ProbNumDiffEq
using LinearAlgebra
using Test

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_pleiades


d = 2
q = 6
D = d*(q+1)


a, b = 1.1, -0.5
f(u, p, t) = [a*u[1], b*u[2]]
u0 = [0.1, 1.0]
tspan = (0.0, 5.0)
t0, T = tspan
prob = ODEProblem(f, u0, tspan)
p = prob.p

# True Solutions and derivatives
u(t) = [a^0*u0[1] * exp(a*t), u0[2] * exp(b*t)]
du(t) = [a^1*u0[1] * exp(a*t), b * u0[2] * exp(b*t)]
ddu(t) = [a^2*u0[1] * exp(a*t), (b)^2 * u0[2] * exp(b*t)]
dddu(t) = [a^3*u0[1] * exp(a*t), (b)^3 * u0[2] * exp(b*t)]
ddddu(t) = [a^4*u0[1] * exp(a*t), (b)^4 * u0[2] * exp(b*t)]
dddddu(t) = [a^5*u0[1] * exp(a*t), (b)^5 * u0[2] * exp(b*t)]
ddddddu(t) = [a^6*u0[1] * exp(a*t), (b)^6 * u0[2] * exp(b*t)]
true_init_states = [u(t0); du(t0); ddu(t0); dddu(t0); ddddu(t0); dddddu(t0); ddddddu(t0)]


@testset "OOP state init" begin
    dfs = ProbNumDiffEq.get_derivatives(prob.u0, prob.f, prob.p, prob.tspan[1], q)
    @test length(dfs) == q+1
    @test true_init_states ≈ vcat(dfs...)
end


@testset "IIP state init" begin
    f!(du, u, p, t) = (du .= f(u, p, t))
    prob = ODEProblem(f!, u0, tspan)

    dfs = ProbNumDiffEq.get_derivatives(prob.u0, prob.f, prob.p, prob.tspan[1], q)
    @test length(dfs) == q+1
    @test true_init_states ≈ vcat(dfs...)
end


@testset "RK-Init on the high-dimensional Pleiades problem" begin
	  # This has not worked before with the Taylor-Mode init!
    # The high dimensions made the runtimes explode

    prob = prob_ode_pleiades
    d = length(prob.u0)

    integ = init(prob, EK0(order=3));
    tm_init = integ.cache.x.μ

    for o in (4, 5)
        integ = init(prob, EK0(order=o, initialization=RungeKuttaInit()));
        rk_init = integ.cache.x.μ

        @test tm_init[1:d] ≈ rk_init[1:d]
        @test tm_init[d+1:2d] ≈ rk_init[d+1:2d]
        @test tm_init[2d+1:3d] ≈ rk_init[2d+1:3d] rtol=1e-2
        @test tm_init[3d+1:4d] ≈ rk_init[3d+1:4d] rtol=1e-1
    end
end
