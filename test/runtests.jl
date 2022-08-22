using Test, SafeTestsets, Aqua
using ProbNumDiffEq
using ODEProblemLibrary
using TimerOutputs
const to = TimerOutput()
macro timedtestset(name, code)
    return esc(:(println("Start testset: ", $name);
    @timeit to $name @safetestset $name $code;
    println("Done.")))
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

    @timedtestset "DE-stats" begin
        include("destats.jl")
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

    @timedtestset "Automatic Differentiation" begin
        include("autodiff.jl")
    end

    @timedtestset "Second order ODEs" begin
        include("secondorderode.jl")
    end

    @timedtestset "Specific Problems" begin
        include("specific_problems.jl")
    end

    @timedtestset "DiffEqDevTools.jl Compatibility" begin
        include("diffeqdevtools.jl")
    end

    @testset "Aqua.jl" begin
        Aqua.test_all(ProbNumDiffEq, ambiguities=false)
        # Aqua.test_ambiguities(ProbNumDiffEq)
    end
end

println("\nTimer Summary:")
display(to)
println("\n")
