using Test
using ProbNumDiffEq
using Statistics
using UnPack
using Plots
using DiffEqDevTools
using ForwardDiff
using ParameterizedFunctions

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary:
    prob_ode_linear,
    prob_ode_2Dlinear,
    prob_ode_lotkavoltera,
    prob_ode_fitzhughnagumo,
    prob_ode_vanstiff,
    prob_ode_mm_linear

using TimerOutputs
const to = TimerOutput()
macro timedtestset(name, code)
    return esc(:(
        println("Start testset: ", $name);
        @timeit to $name @testset $name $code;
        println("Done.");
    ))
end

@testset "ProbNumDiffEq" begin
    @timedtestset "Correctness" begin
        include("correctness.jl")
    end

    @timedtestset "Filtering" begin
        include("filtering.jl")
    end

    @timedtestset "Convergence" begin
        include("convergence.jl")
    end

    @timedtestset "Priors" begin
        include("priors.jl")
    end

    @timedtestset "Solution" begin
        include("solution.jl")
    end

    @timedtestset "Diffusions" begin
        include("diffusions.jl")
    end

    @timedtestset "State Initialization" begin
        include("state_init.jl")
    end

    @timedtestset "Preconditioning" begin
        include("preconditioning.jl")
    end

    @timedtestset "Smoothing" begin
        include("smoothing.jl")
    end

    @timedtestset "Errors" begin
        include("errors.jl")
    end

    @timedtestset "IEKS" begin
        include("ieks.jl")
    end

    @timedtestset "Automatic Differentiation" begin
        include("autodiff.jl")
    end

    @timedtestset "Specific Problems" begin
        include("specific_problems.jl")
    end
end

println("\nTimer Summary:")
display(to)
println("\n")
