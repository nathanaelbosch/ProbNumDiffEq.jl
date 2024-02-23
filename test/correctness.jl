# Goal: Make sure that our solvers are "correct" for small steps or tolerances
# Verify this for many (ideally all) combinations
# Compare with an algorithm from OrdinaryDiffEq.jl with high precision
using ProbNumDiffEq
using Test
using OrdinaryDiffEq
using LinearAlgebra
using Statistics: mean
using DiffEqDevTools
import ODEProblemLibrary: prob_ode_lotkavolterra, prob_ode_fitzhughnagumo

CONSTANT_ALGS = (
    EK0(order=2, smooth=false) => 1e-5,
    EK0(order=3, smooth=false) => 1e-8,
    EK0(order=5, smooth=false) => 1e-10,
    EK0(order=3, smooth=false, diffusionmodel=FixedDiffusion()) => 1e-7,
    EK0(order=3, smooth=false, diffusionmodel=FixedMVDiffusion()) => 1e-7,
    EK0(order=3, smooth=false, diffusionmodel=DynamicMVDiffusion()) => 1e-8,
    EK0(order=3, smooth=false, initialization=ClassicSolverInit()) => 1e-7,
    EK0(order=3, smooth=false, initialization=SimpleInit()) => 1e-5,
    EK0(
        order=3,
        smooth=false,
        diffusionmodel=FixedMVDiffusion(),
        initialization=ClassicSolverInit(),
    ) => 1e-7,
    EK1(order=2, smooth=false) => 1e-7,
    EK1(order=3, smooth=false) => 1e-8,
    EK1(order=5, smooth=false) => 1e-11,
    EK1(order=3, smooth=false, diffusionmodel=FixedDiffusion()) => 1e-8,
    EK1(order=3, smooth=false, initialization=ClassicSolverInit()) => 1e-7,
    EK1(order=3, smooth=false, initialization=SimpleInit()) => 1e-5,
    # DiagonalEK1(order=3, smooth=false) => 1e-7,
    # DiagonalEK1(order=3, smooth=false, diffusionmodel=FixedDiffusion()) => 1e-7,
    # DiagonalEK1(order=3, smooth=false, diffusionmodel=DynamicDiffusion()) => 1e-7,
    # DiagonalEK1(order=3, smooth=false, initialization=ClassicSolverInit()) => 1e-7,
    # smoothing
    EK0(order=3, smooth=true) => 1e-8,
    EK0(order=3, smooth=true, diffusionmodel=FixedDiffusion()) => 2e-8,
    EK0(order=3, smooth=true, diffusionmodel=FixedMVDiffusion()) => 1e-7,
    EK0(order=3, smooth=true, diffusionmodel=DynamicMVDiffusion()) => 1e-8,
    EK1(order=3, smooth=true) => 1e-8,
    EK1(order=3, smooth=true, diffusionmodel=FixedDiffusion()) => 1e-8,
    # DiagonalEK1(order=3, smooth=true) => 1e-7,
    # DiagonalEK1(order=3, smooth=true, diffusionmodel=FixedDiffusion()) => 1e-7,
    # DiagonalEK1(order=3, smooth=true, diffusionmodel=DynamicDiffusion()) => 1e-7,
    # DiagonalEK1(order=3, smooth=true, initialization=ClassicSolverInit()) => 1e-7,
    # Priors
    EK0(prior=IOUP(3, -1), smooth=true) => 2e-9,
    EK1(prior=IOUP(3, -1), smooth=true, diffusionmodel=FixedDiffusion()) => 1e-9,
    EK1(prior=IOUP(3, update_rate_parameter=true), smooth=true) => 1e-9,
    EK0(prior=Matern(3, 1), smooth=true) => 5e-7,
    EK1(prior=Matern(4, 0.1), smooth=true, diffusionmodel=FixedDiffusion()) => 2e-5,
)
ADAPTIVE_ALGS = (
    EK0(order=2) => 2e-4,
    EK0(order=3) => 1e-4,
    EK0(order=5) => 1e-5,
    EK0(order=8) => 2e-5,
    EK0(order=3, diffusionmodel=DynamicMVDiffusion()) => 5e-5,
    EK0(order=3, initialization=ClassicSolverInit()) => 5e-5,
    EK0(order=3, initialization=SimpleInit()) => 1e-4,
    EK0(order=3, diffusionmodel=DynamicMVDiffusion(), initialization=ClassicSolverInit()) => 4e-5,
    EK0(order=3, diffusionmodel=FixedMVDiffusion()) => 1e-4,
    EK1(order=2) => 2e-5,
    EK1(order=3) => 1e-5,
    EK1(order=5) => 1e-6,
    EK1(order=8) => 5e-6,
    EK1(order=3, initialization=ClassicSolverInit()) => 1e-5,
    EK1(order=3, initialization=SimpleInit()) => 1e-4,
    # DiagonalEK1(order=3) => 1e-4,
    # DiagonalEK1(order=3, diffusionmodel=FixedDiffusion()) => 1e-4,
    # DiagonalEK1(order=3, diffusionmodel=DynamicDiffusion()) => 1e-4,
    # DiagonalEK1(order=3, initialization=ClassicSolverInit()) => 1e-4,
    # Priors
    EK0(prior=IOUP(3, -1), smooth=true) => 1e-5,
    EK1(prior=IOUP(3, -1), smooth=true, diffusionmodel=FixedDiffusion()) => 1e-5,
    EK1(prior=IOUP(3, update_rate_parameter=true), smooth=true) => 2e-5,
    EK0(prior=Matern(3, 1), smooth=true) => 1e-4,
    EK1(prior=Matern(3, 0.1), smooth=true, diffusionmodel=FixedDiffusion()) => 1e-5,
)

PROBS = (
    (prob_ode_lotkavolterra, "lotkavolterra"),
    (prob_ode_fitzhughnagumo, "fitzhughnagumo"),
)

for (prob, probname) in PROBS
    true_sol = solve(prob, Vern9(), abstol=1e-12, reltol=1e-12)
    testsol = TestSolution(true_sol)

    @testset "Constant steps: $probname; alg=$alg" for (alg, err) in CONSTANT_ALGS
        sol = solve(
            prob,
            alg,
            adaptive=false,
            dt=1e-2,
            dense=alg.smooth,
            save_everystep=alg.smooth,
        )
        appxsol = appxtrue(sol, true_sol, dense_errors=false)
        @test appxsol.errors[:final] < err
        # @test appxsol.errors[:l2] < err
        # @test appxsol.errors[:L2] < err
    end

    @testset "Adaptive: $probname; alg=$alg" for (alg, err) in ADAPTIVE_ALGS
        sol = solve(prob, alg)
        appxsol = appxtrue(sol, testsol)
        @test appxsol.errors[:final] < err
        @test appxsol.errors[:l2] < err
        @test appxsol.errors[:L2] < err
    end
end
