using ProbNumDiffEq
using Test
using Statistics
using UnPack
using Plots
using DiffEqDevTools
using ForwardDiff
using ParameterizedFunctions

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo, prob_ode_vanstiff, prob_ode_mm_linear

using TimerOutputs
const to = TimerOutput()
macro timedtestset(name, code)
    return esc(:(@timeit to $name @testset $name $code))
end

@testset "ProbNumDiffEq" begin

    println("Correctness")
    @timedtestset "Correctness" begin include("correctness.jl") end

    println("Filtering")
    @timedtestset "Filtering" begin include("filtering.jl") end

    println("Convergence")
    @timedtestset "Convergence" begin include("convergence.jl") end

    println("Priors")
    @timedtestset "Priors" begin include("priors.jl") end

    println("Solution")
    @timedtestset "Solution" begin include("solution.jl") end

    println("Diffusions")
    @timedtestset "Diffusions" begin include("diffusions.jl") end

    println("State Initialization")
    @timedtestset "State Initialization" begin include("state_init.jl") end

    println("Preconditioning")
    @timedtestset "Preconditioning" begin include("preconditioning.jl") end

    println("Smoothing")
    @timedtestset "Smoothing" begin include("smoothing.jl") end

    println("Errors")
    @timedtestset "Errors" begin include("errors.jl") end

    println("IEKS")
    @timedtestset "IEKS" begin include("ieks.jl") end

    println("Automatic Differentiation")
    @timedtestset "Automatic Differentiation" begin include("autodiff.jl") end

    println("Specific Problems")
    @timedtestset "Specific Problems" begin include("specific_problems.jl") end
end

println("\nTimer Summary:")
display(to)
println("\n")
