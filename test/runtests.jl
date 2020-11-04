using Test
using ODEFilters


@testset "ODEFilters" begin

    println("Correctness")
    @testset "Correctness" begin include("correctness.jl") end

    println("Priors")
    @testset "Priors" begin include("priors.jl") end

    println("Solution")
    @testset "Solution" begin include("solution.jl") end

    println("Diffusions")
    @testset "Diffusions" begin include("diffusions.jl") end

    println("State Initialization")
    @testset "State Initialization" begin include("state_init.jl") end

    println("Preconditioning")
    @testset "Preconditioning" begin include("preconditioning.jl") end

    println("Smoothing")
    @testset "Smoothing" begin include("smoothing.jl") end

    println("Errors")
    @testset "Errors" begin include("errors.jl") end

    println("IEKS")
    @testset "IEKS" begin include("ieks.jl") end

    println("Specific Problems")
    @testset "Specific Problems" begin include("specific_problems.jl") end
end
