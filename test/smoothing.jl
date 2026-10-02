using ProbNumDiffEq
using Test
using LinearAlgebra
using OrdinaryDiffEq
using OrdinaryDiffEqFIRK: RadauIIA5
using Statistics: mean
import ODEProblemLibrary: prob_ode_lotkavolterra
using Plots

prob = prob_ode_lotkavolterra

@testset "Smoothing for small dt and large q" begin
    dt = 1e-5
    q = 8
    @test_nowarn solve(
        remake(prob, tspan=(0.0, 10dt)),
        EK0(order=q, smooth=true, diffusionmodel=FixedDiffusion()),
        adaptive=false,
        dt=dt,
    )
    @test_nowarn solve(
        remake(prob, tspan=(0.0, 10dt)),
        EK1(order=q, smooth=true, diffusionmodel=FixedDiffusion()),
        adaptive=false,
        dt=dt,
    )
end

@testset "Stable smoothing at high order on a stiff problem (#393)" begin
    function vanderpol!(du, u, p, t)
        du[1] = u[2]
        du[2] = p[1] * ((1 - u[1]^2) * u[2] - u[1])
    end
    vdp = ODEProblem(vanderpol!, [2.0, 0.0], (0.0, 2.0), [1e5])
    ref = solve(vdp, RadauIIA5(), abstol=1e-12, reltol=1e-12)
    ts = range(vdp.tspan..., length=200)
    @testset "order $order" for order in (6, 7)
        sol = solve(vdp, EK1(; order, smooth=true), abstol=1e-10, reltol=1e-7)
        @test maximum(norm(mean(sol(t)) - ref(t)) for t in ts) < 1e-4
    end
end

@testset "Smooth vs. non-smooth" begin
    q = 3
    dt = 1e-2

    sol_nonsmooth =
        solve(prob, EK0(order=q, smooth=false), dense=false, adaptive=false, dt=dt)
    sol_smooth = solve(prob, EK0(order=q, smooth=true), adaptive=false, dt=dt)

    @test sol_nonsmooth.t ≈ sol_smooth.t
    @test sol_nonsmooth.u[end] == sol_smooth.u[end]
    @test sol_nonsmooth.u[end-1] != sol_smooth.u[end-1]

    plot(sol_smooth, label="smooth")
    plot!(sol_nonsmooth, label="nonsmooth")

    sol_best = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
    sol_best_u = sol_best.(sol_smooth.t)

    nonsmooth_errors = norm.(sol_nonsmooth.u - sol_best_u, 2)
    smooth_errors = norm.(sol_smooth.u - sol_best_u, 2)

    @test 2 * maximum(nonsmooth_errors) > maximum(smooth_errors)
    @test 2 * sum(nonsmooth_errors) > sum(smooth_errors)

    # Previously we compared the smooth and non-smooth dense output, but this
    # does not work anymore since dense output requires smoothing!
end
