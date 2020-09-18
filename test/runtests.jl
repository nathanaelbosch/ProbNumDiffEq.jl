using Test
using ProbNumODE


@testset "ProbNumODE" begin

    println("Correctness")
    @testset "Correctness" begin include("correctness.jl") end

    println("Priors")
    @testset "Priors" begin include("priors.jl") end

    println("Solution")
    @testset "Solution" begin include("solution.jl") end

    println("Sigmas")
    @testset "Sigmas" begin include("sigmas.jl") end

    println("State Initialization")
    @testset "State Initialization" begin include("state_init.jl") end

    println("Preconditioning")
    @testset "Preconditioning" begin include("preconditioning.jl") end

    println("Step Control")
    @testset "Step Control" begin include("step_controller.jl") end

    println("Smoothing")
    @testset "Smoothing" begin include("smoothing.jl") end

    println("Errors")
    @testset "Errors" begin include("errors.jl") end

    println("Specific Problems")
    @testset "Specific Problems" begin include("specific_problems.jl") end
end
