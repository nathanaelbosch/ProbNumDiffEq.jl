using Test
using LinearAlgebra
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo
using ModelingToolkit


prob = prob_ode_lotkavoltera
prob = ProbNumODE.remake_prob_with_jac(prob)


@testset "Smoothing for small dt and large q" begin
    dt = 1e-5
    q = 4
    @test solve(
        prob, EKF0(), q=q, smooth=true,
        sigmarule=:schober,
        steprule=:constant,
        dt=dt,
    ) isa DiffEqBase.AbstractODESolution
end


@testset "Smooth vs. non-smooth" begin
    q = 3
    dt = 1e-2
    method = EKF0()

    sol_nonsmooth = solve(prob, method, q=q, steprule=:constant, dt=dt, smooth=false);
    sol_smooth = solve(prob, method, q=q, steprule=:constant, dt=dt, smooth=true);

    @test sol_nonsmooth.t ≈ sol_smooth.t
    @test sol_nonsmooth[end] == sol_smooth[end]
    @test sol_nonsmooth[end-1] != sol_smooth[end-1]

    plot(sol_smooth, label="smooth"); plot!(sol_nonsmooth, label="nonsmooth")

    sol_best = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12);
    sol_best_u = sol_best.(sol_smooth.t);

    nonsmooth_errors = norm.(sol_nonsmooth.u - sol_best_u, 2)
    smooth_errors = norm.(sol_smooth.u - sol_best_u, 2)

    @test 2*maximum(nonsmooth_errors) > maximum(smooth_errors)
    @test 2*sum(nonsmooth_errors) > sum(smooth_errors)
end
