using Test
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo



@testset "Test the different diffusion models" begin
    prob = prob_ode_fitzhughnagumo
    true_sol = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)

    @testset "Time-Varying Diffusion" begin
        sol = solve(prob, EK0(diffusionmodel=:dynamic, smooth=false),
                    dense=false, adaptive=false, dt=1e-4)
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diffusion" begin
        sol = solve(prob, EK0(diffusionmodel=:fixed, smooth=false),
                    dense=false, adaptive=false, dt=1e-4)
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Varying Diagonal Diffusion" begin
        sol = solve(prob, EK0(diffusionmodel=:dynamicMV, smooth=false),
                    dense=false, adaptive=false, dt=1e-4)
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diagonal Diffusion" begin
        sol = solve(prob, EK0(diffusionmodel=:fixedMV, smooth=false),
                    dense=false, adaptive=false, dt=1e-4)
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

    @testset "Time-Fixed Diffusion MAP" begin
        sol = solve(prob, EK0(diffusionmodel=:fixedMAP, smooth=false),
                    dense=false, adaptive=false, dt=1e-4)
        @test sol.u[end] ≈ true_sol.(sol.t)[end]
    end

end
