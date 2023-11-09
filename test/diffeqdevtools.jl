using Test
using ProbNumDiffEq
using OrdinaryDiffEq
using DiffEqDevTools
using Plots

import ODEProblemLibrary: prob_ode_fitzhughnagumo
prob = prob_ode_fitzhughnagumo

ref = solve(prob, Vern7(), abstol=1 / 10^14, reltol=1 / 10^14)
test_sol = TestSolution(ref)
appxsol_nondense =
    solve(prob, Vern7(), abstol=1 / 10^14, reltol=1 / 10^14, dense=false)
test_sol_nondense = TestSolution(appxsol_nondense)

@testset "appxtrue" begin
    @testset "$S" for (S, _sol) in (("TestSolution", test_sol), ("ODESolution", ref))
        appxsol = appxtrue(solve(prob, EK1()), _sol)
        @test appxsol.errors isa Dict
        @test :final in keys(appxsol.errors)
        @test :l2 in keys(appxsol.errors)
        @test :L2 in keys(appxsol.errors)
        @test :l∞ in keys(appxsol.errors)
        @test :L∞ in keys(appxsol.errors)

        sol = solve(prob, EK1(smooth=false), dense=false)
        appxsol = appxtrue(sol, _sol)
        @test appxsol.errors isa Dict
        @test :final in keys(appxsol.errors)
        @test :l2 in keys(appxsol.errors)
        @test :l∞ in keys(appxsol.errors)

        appxsol = appxtrue(sol, test_sol_nondense)
        @test appxsol.errors isa Dict
        @test :final in keys(appxsol.errors)
        @test :l2 in keys(appxsol.errors)
        @test :l∞ in keys(appxsol.errors)
    end
end

@testset "WorkPrecision" begin
    abstols = 1.0 ./ 10.0 .^ (6:7)
    reltols = 1.0 ./ 10.0 .^ (3:4)
    wp = WorkPrecision(
        prob,
        EK1(smooth=false),
        abstols,
        reltols;
        appxsol=test_sol,
        dense=false,
        maxiters=100000,
        numruns=2,
    )
    @test wp isa WorkPrecision
    @test plot(wp) isa AbstractPlot
end

@testset "WorkPrecisionSet" begin
    abstols = 1.0 ./ 10.0 .^ (6:7)
    reltols = 1.0 ./ 10.0 .^ (3:4)
    setups = [
        Dict(:alg => EK0())
        Dict(:alg => EK1())
    ]
    wps = WorkPrecisionSet(
        prob,
        abstols,
        reltols,
        setups;
        appxsol=test_sol,
        error_estimate=:L2,
        maxiters=100000,
        numruns=2,
    )
    @test wps isa WorkPrecisionSet
    @test plot(wps) isa AbstractPlot
end

@testset "WorkPrecisionSet with TestSolution is broken" begin
    abstols = 1.0 ./ 10.0 .^ (6:7)
    reltols = 1.0 ./ 10.0 .^ (3:4)
    setups = [
        Dict(:alg => EK0(smooth=false))
        Dict(:alg => EK1(smooth=false))
    ]
    @test_broken wp = WorkPrecisionSet(
        prob, abstols, reltols, setups;
        appxsol=test_sol,
        dense=false,
        save_everystep=false,
        numruns=2,
        maxiters=Int(1e7),
        timeseries_errors=false,
        verbose=false,
    )
    @test_nowarn WorkPrecisionSet(
        prob, abstols, reltols, setups;
        appxsol=appxsol_nondense,
        dense=false,
        save_everystep=false,
        numruns=2,
        maxiters=Int(1e7),
        timeseries_errors=false,
        verbose=false,
    )
end
