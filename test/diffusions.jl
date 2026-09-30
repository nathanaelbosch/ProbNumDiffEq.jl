using Test
using ProbNumDiffEq
using OrdinaryDiffEq
using DiffEqDevTools
using LinearAlgebra
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
        @test length(sol_end(0.5).μ) == length(prob.u0)
    end

    @testset "`FixedMVDiffusion` calibrates each dimension with its own `S[j, j]`" begin
        @testset "`DiagonalEK1` on a decoupled problem matches each 1-d problem" begin
            λ = [-1.0, -100.0]
            f(du, u, p, t) = (du .= λ .* u)
            kwargs = (adaptive=false, dt=1e-2)
            prob = ODEProblem(f, [1.0, 1.0], (0.0, 1.0))
            sol = solve(prob, DiagonalEK1(diffusionmodel=FixedMVDiffusion()); kwargs...)
            @test !(sol.diffusions[end][1, 1] ≈ sol.diffusions[end][2, 2])
            for j in 1:2
                prob_j = ODEProblem((du, u, p, t) -> (du .= λ[j] .* u), [1.0], (0.0, 1.0))
                sol_j =
                    solve(prob_j, DiagonalEK1(diffusionmodel=FixedDiffusion()); kwargs...)
                @test sol.diffusions[end][j, j] ≈ sol_j.diffusions[end]
                @test Matrix(sol.pu[end].Σ)[j, j] ≈ Matrix(sol_j.pu[end].Σ)[1, 1]
            end
        end

        @testset "`EK0` with a diagonal mass matrix matches the problem without it" begin
            m = [1.0, 10.0]
            g(du, u, p, t) = (du .= [-1.0, -2.0] .* u)
            gM(du, u, p, t) = (du .= m .* [-1.0, -2.0] .* u)
            probI = ODEProblem(g, [1.0, 1.0], (0.0, 1.0))
            probM =
                ODEProblem(ODEFunction(gM; mass_matrix=Diagonal(m)), [1.0, 1.0], (0.0, 1.0))
            # `SimpleInit` gives both problems the same initial state
            alg = EK0(diffusionmodel=FixedMVDiffusion(), initialization=SimpleInit())
            kwargs = (adaptive=false, dt=1e-2)
            solI = solve(probI, alg; kwargs...)
            solM = solve(probM, alg; kwargs...)
            @test solM.diffusions[end] ≈ solI.diffusions[end]
            @test Matrix(solM.pu[end].Σ) ≈ Matrix(solI.pu[end].Σ)
        end
    end
end
