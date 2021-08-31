# Goal: Test the correctness of state initialization
using ProbNumDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Test

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_pleiades


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
    dfs = ProbNumDiffEq.taylormode_get_derivatives(prob.u0, prob.f, prob.p, prob.tspan[1], q)
    @test length(dfs) == q+1
    @test true_init_states ≈ vcat(dfs...)
end


@testset "IIP state init" begin
    f!(du, u, p, t) = (du .= f(u, p, t))
    prob = ODEProblem(f!, u0, tspan)

    dfs = ProbNumDiffEq.taylormode_get_derivatives(prob.u0, prob.f, prob.p, prob.tspan[1], q)
    @test length(dfs) == q+1
    @test true_init_states ≈ vcat(dfs...)
end



@testset "Compare TaylorModeInit and RungeKuttaInit" begin
	  # This has not worked before with the Taylor-Mode init!
    # The high dimensions made the runtimes explode

    prob = prob_ode_fitzhughnagumo
    d = length(prob.u0)

    integ = init(prob, EK0(order=8));
    tm_init = integ.cache.x.μ

    @testset "Order $o" for o in (3, 4, 5, 6, 7)
        integ = init(prob, EK0(order=o, initialization=RungeKuttaInit()));
        rk_init = integ.cache.x.μ

        # These are fit via the initial values + autodiff, and should be good
        @test tm_init[1:d] ≈ rk_init[1:d]
        @test tm_init[d+1:2d] ≈ rk_init[d+1:2d]
        (o > 1) && @test tm_init[2d+1:3d] ≈ rk_init[2d+1:3d]

        # Test if the others are correct, up to order 5
        (5 > o > 2) && @test tm_init[3d+1:4d] ≈ rk_init[3d+1:4d] rtol=1e-2
        (o == 5) && @test tm_init[3d+1:4d] ≈ rk_init[3d+1:4d] rtol=6e-1
        (5 > o > 3) && @test tm_init[4d+1:5d] ≈ rk_init[4d+1:5d] rtol=1e-3
        (o == 5) && @test tm_init[4d+1:5d] ≈ rk_init[4d+1:5d] rtol=1e-2
        (o == 5) && @test tm_init[5d+1:6d] ≈ rk_init[5d+1:6d] rtol=2e-1


        # Test if the covariance reflects the true error
        Cs = integ.cache.x.Σ.squareroot[3d+1:end, 3d+1:end]
        err = (rk_init .- tm_init[1:length(rk_init)])[3d+1:end]
        whitened_err = Cs \ err
        # `whitened_err` should be standard gaussian; so, let's check that they're small
        @test all(abs.(whitened_err) .< 2e-1)

    end
end


@testset "RK-Init enables solving the high-dimensional Pleiades problem with order 5" begin
	  # This has not worked before with the Taylor-Mode init, it simply took too long!
    prob = prob_ode_pleiades
    sol1 = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10, save_everystep=false)
    sol2 = solve(prob, EK1(order=5, smooth=false, initialization=RungeKuttaInit()),
                 abstol=1e-6, reltol=1e-6, save_everystep=false)
    @test sol1.u[end] ≈ sol2.u[end] rtol=2e-6
end
