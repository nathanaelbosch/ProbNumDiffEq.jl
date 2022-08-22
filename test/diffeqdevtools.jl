using Test
using ProbNumDiffEq
using OrdinaryDiffEq
using DiffEqDevTools
using Plots

import ODEProblemLibrary: prob_ode_fitzhughnagumo
prob = prob_ode_fitzhughnagumo

test_sol = TestSolution(solve(prob, Vern7(), abstol=1 / 10^14, reltol=1 / 10^14))
test_sol_nondense =
    TestSolution(solve(prob, Vern7(), dense=false, abstol=1 / 10^14, reltol=1 / 10^14))

@testset "appxtrue" begin
    appxsol = appxtrue(solve(prob, EK1()), test_sol)
    @test appxsol.errors isa Dict
    @test :final in keys(appxsol.errors)
    @test :l2 in keys(appxsol.errors)
    @test :L2 in keys(appxsol.errors)
    @test :l∞ in keys(appxsol.errors)
    @test :L∞ in keys(appxsol.errors)

    sol = solve(prob, EK1(smooth=false), dense=false)
    appxsol = appxtrue(sol, test_sol)
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

@testset "WorkPrecision" begin
    abstols = 1.0 ./ 10.0 .^ (6:13)
    reltols = 1.0 ./ 10.0 .^ (3:10)
    wp = WorkPrecision(
        prob,
        EK1(smooth=false),
        abstols,
        reltols;
        appxsol=test_sol,
        dense=false,
        maxiters=100000,
        numruns=10,
    )
    @test wp isa WorkPrecision
    @test plot(wp) isa AbstractPlot
end

@testset "WorkPrecisionSet" begin
    abstols = 1.0 ./ 10.0 .^ (6:13)
    reltols = 1.0 ./ 10.0 .^ (3:10)
    setups = [
        Dict(:alg => EK0(order=4))
        Dict(:alg => EK1(order=5))
    ]
    wps = WorkPrecisionSet(
        prob,
        abstols,
        reltols,
        setups;
        appxsol=test_sol,
        error_estimate=:L2,
        maxiters=100000,
        numruns=10,
    )
    @test wps isa WorkPrecisionSet
    @test plot(wps) isa AbstractPlot
end
