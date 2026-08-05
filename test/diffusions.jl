using Test
using ProbNumDiffEq
using OrdinaryDiffEq
using DiffEqDevTools
import ODEProblemLibrary: prob_ode_fitzhughnagumo

@testset "Test the different diffusion models" begin
    prob = prob_ode_fitzhughnagumo
    true_sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)

    @testset "Time-Varying Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=DynamicDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < 1e-5
    end

    @testset "Time-Fixed Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < 1e-5
    end

    @testset "Time-Fixed Diffusion - uncalibrated and with custom initial value" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedDiffusion(1e3, false), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < 1e-5
    end

    @testset "Time-Varying Diagonal Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=DynamicMVDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < 1e-5
    end

    @testset "Time-Fixed Diagonal Diffusion" begin
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedMVDiffusion(), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < 1e-5
    end

    @testset "Time-Fixed Diagonal Diffusion - uncalibrated and with custom values" begin
        d = length(prob.u0)
        initial_diffusion = 1 .+ rand(d)
        sol = solve(
            prob,
            EK0(diffusionmodel=FixedMVDiffusion(initial_diffusion, false), smooth=false),
            dense=false,
            adaptive=false,
            dt=1e-3,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < 1e-5
    end

    # Fixes https://github.com/nathanaelbosch/ProbNumDiffEq.jl/issues/428
    @testset "`save_everystep=false` returns the same endpoint: $D" for D in (
        FixedDiffusion(),
        FixedMVDiffusion(),
        FixedDiffusion(1e3, false),
        DynamicDiffusion(),
        DynamicMVDiffusion(),
    )
        alg = EK0(diffusionmodel=D, smooth=false)
        kwargs = (dense=false, adaptive=false, dt=1e-2)
        sol_all = solve(prob, alg; save_everystep=true, kwargs...)
        sol_end = solve(prob, alg; save_everystep=false, kwargs...)

        @test length(sol_end.u) == length(sol_end.pu) == length(sol_end.x_filt) == 2
        @test length(sol_end.diffusions) == 1

        @test sol_end.pu[end].μ ≈ sol_all.pu[end].μ
        @test Matrix(sol_end.pu[end].Σ) ≈ Matrix(sol_all.pu[end].Σ)
        @test sol_end.diffusions[end] ≈ sol_all.diffusions[end]

        # the interpolation reads `sol.diffusions`, which used to be empty here
        @test length(sol_end(0.5).μ) == length(prob.u0)
    end
end
