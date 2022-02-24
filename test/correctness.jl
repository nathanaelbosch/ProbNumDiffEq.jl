# Goal: Make sure that our solvers are "correct" for small steps or tolerances
# Verify this for many (ideally all) combinations
# Compare with an algorithm from OrdinaryDiffEq.jl with high precision
using ProbNumDiffEq
using Test
using OrdinaryDiffEq
using LinearAlgebra
using Statistics: mean
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary:
    prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

EK1FDB1(; kwargs...) = EK1FDB(; jac_quality=1, kwargs...)
EK1FDB2(; kwargs...) = EK1FDB(; jac_quality=2, kwargs...)
EK1FDB3(; kwargs...) = EK1FDB(; jac_quality=3, kwargs...)
# ALGS = [EK0, EK1, EK1FDB1, EK1FDB2, EK1FDB3]
ALGS = [EK0, EK1]
DIFFUSIONS = [FixedDiffusion, DynamicDiffusion, FixedMVDiffusion, DynamicMVDiffusion]
INITS = [TaylorModeInit, ClassicSolverInit]

for (prob, probname) in
    [(prob_ode_lotkavoltera, "lotkavolterra"), (prob_ode_fitzhughnagumo, "fitzhughnagumo")]
    @testset "Constant steps: $probname" begin
        true_sol =
            solve(remake(prob, u0=big.(prob.u0)), Tsit5(), abstol=1e-20, reltol=1e-20)

        for Alg in ALGS, diffusion in DIFFUSIONS, init in INITS, q in [2, 3, 5]
            if (diffusion == FixedMVDiffusion || diffusion == DynamicMVDiffusion) &&
               Alg != EK0
                continue
            end
            if init == ClassicSolverInit && q > 4
                continue
            end

            @testset "Constant steps: $probname; alg=$Alg, diffusion=$diffusion, init=$init, q=$q" begin
                @debug "Testing for correctness: Constant steps" probname alg diffusion q dt

                sol = solve(
                    prob,
                    Alg(order=q, diffusionmodel=diffusion(), initialization=init()),
                    adaptive=false,
                    dt=1e-2,
                )
                @test sol.u ≈ true_sol.(sol.t) rtol = 1e-5
            end
        end
    end
end

for (prob, probname) in
    [(prob_ode_lotkavoltera, "lotkavolterra"), (prob_ode_fitzhughnagumo, "fitzhughnagumo")]
    @testset "Adaptive steps: $probname" begin
        t_eval = prob.tspan[1]:0.01:prob.tspan[end]
        true_sol =
            solve(remake(prob, u0=big.(prob.u0)), Tsit5(), abstol=1e-20, reltol=1e-20)
        true_dense_vals = true_sol.(t_eval)

        for Alg in ALGS, diffusion in DIFFUSIONS, init in INITS, q in [2, 3, 5]
            if (diffusion == FixedMVDiffusion || diffusion == DynamicMVDiffusion) &&
               Alg != EK0
                continue
            end

            @testset "Adaptive steps: $probname; alg=$Alg, diffusion=$diffusion, init=$init, q=$q" begin
                @debug "Testing for correctness: Adaptive steps" probname Alg diffusion q

                sol = solve(
                    prob,
                    Alg(order=q, diffusionmodel=diffusion(), initialization=init()),
                    adaptive=true,
                )

                @test sol.u ≈ true_sol.(sol.t) rtol = 1e-3
                @test mean.(sol.(t_eval)) ≈ true_dense_vals rtol = 1e-3
            end
        end
    end
end
