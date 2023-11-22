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

const GROUP = get(ENV, "GROUP", "All")

@testset "ProbNumDiffEq" begin
    if GROUP == "All" || GROUP == "Core"
        @timedtestset "Core" begin
            @timedsafetestset "Filtering" begin
                include("core/filtering.jl")
            end
            @timedsafetestset "Priors" begin
                include("core/priors.jl")
            end
            @timedsafetestset "Preconditioning" begin
                include("core/preconditioning.jl")
            end
            @timedsafetestset "Measurement Models" begin
                include("core/measurement_models.jl")
            end
            #
            @timedsafetestset "State Initialization" begin
                include("state_init.jl")
            end
            @timedsafetestset "Smoothing" begin
                include("smoothing.jl")
            end
            @timedsafetestset "IsometricKroneckerProduct" begin
                include("core/kronecker.jl")
            end
        end
    end

    if GROUP == "All" || GROUP == "Downstream" || GROUP == "Solvers"
        @timedtestset "Solver Correctness" begin
            @timedsafetestset "Correctness" begin
                include("correctness.jl")
            end
            @timedsafetestset "Convergence" begin
                include("convergence.jl")
            end
            @timedsafetestset "Complexity" begin
                include("complexity.jl")
            end
            @timedsafetestset "Stiff Problem" begin
                include("stiff_problem.jl")
            end
            @timedtestset "Test all diffusion models" begin
                include("diffusions.jl")
            end
            @timedtestset "IOUP" begin
                include("ioup.jl")
            end
            @timedtestset "Exponential Integrators" begin
                include("exponential_integrators.jl")
            end
        end
    end

    if GROUP == "All" || GROUP == "Downstream" || GROUP == "Interface"
        @timedtestset "Interface" begin
            @timedsafetestset "Solution" begin
                include("solution.jl")
            end
            @timedsafetestset "DE-stats" begin
                include("stats.jl")
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
            @timedsafetestset "OOP Problem Compatibility" begin
                include("oop_problems.jl")
            end
            @timedsafetestset "Mass Matrix" begin
                include("mass_matrix.jl")
            end
            @timedsafetestset "ParameterizedFunctions.jl" begin
                include("parameterized_functions.jl")
            end
            @timedsafetestset "Callbacks.jl" begin
                include("callbacks.jl")
            end
            @timedsafetestset "BigFloat" begin
                include("bigfloat.jl")
            end
            @timedsafetestset "Problem with analytic solution" begin
                include("analytic_solution.jl")
            end
            @timedsafetestset "Matrix-valued problem" begin
                include("matrix_valued_problem.jl")
            end
            @timedsafetestset "Scalar-valued problem (broken)" begin
                include("scalar_valued_problem.jl")
            end
            @timedsafetestset "Implicit solver kwarg compat" begin
                include("implicit_solver_kwarg_compat.jl")
            end
        end
    end

    if GROUP == "All"
        @timedtestset "Aqua.jl" begin
            Aqua.test_all(
                ProbNumDiffEq,
                ambiguities=false,
                piracies=false,
            )
        end
    end
end

println("\nTimer Summary:")
display(to)
println("\n")
