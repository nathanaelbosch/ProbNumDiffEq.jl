using Test, SafeTestsets, Aqua, TimerOutputs
using ProbNumDiffEq
using ODEProblemLibrary

to = TimerOutput()
macro timedtestset(name, code)
    return esc(:(println("Start testset: ", $name);
    @timeit to $name @testset $name $code;
    println("Done.")))
end
macro timedsafetestset(name, code)
    return esc(:(println("Start testset: ", $name);
    @timeit to $name @safetestset $name $code;
    println("Done.")))
end

@testset "ProbNumDiffEq" begin
    @timedtestset "Core" begin
        @timedsafetestset "Filtering" begin
            include("filtering.jl")
        end
        @timedsafetestset "Priors" begin
            include("priors.jl")
        end
        @timedsafetestset "Preconditioning" begin
            include("preconditioning.jl")
        end
    end

    @timedtestset "Solver Correctness" begin
        @timedsafetestset "Correctness" begin
            include("correctness.jl")
        end
        @timedsafetestset "Convergence" begin
            include("convergence.jl")
        end
        @timedsafetestset "State Initialization" begin
            include("state_init.jl")
        end
        @timedsafetestset "Smoothing" begin
            include("smoothing.jl")
        end
    end

    @timedtestset "Interface" begin
        @timedsafetestset "Solution" begin
            include("solution.jl")
        end
        @timedsafetestset "DE-stats" begin
            include("destats.jl")
        end
        @timedsafetestset "Errors Thrown" begin
            include("errors_thrown.jl")
        end
        @timedsafetestset "Automatic Differentiation" begin
            include("autodiff.jl")
        end
        @timedsafetestset "Second order ODEs" begin
            include("secondorderode.jl")
        end
        @timedsafetestset "DiffEqDevTools.jl Compatibility" begin
            include("diffeqdevtools.jl")
        end
    end

    @timedtestset "Diffusions" begin
        include("diffusions.jl")
    end

    @timedtestset "Specific Problems" begin
        include("specific_problems.jl")
    end

    @testset "Aqua.jl" begin
        Aqua.test_all(ProbNumDiffEq, ambiguities=false)
        # Aqua.test_ambiguities(ProbNumDiffEq)
    end
end

println("\nTimer Summary:")
display(to)
println("\n")
