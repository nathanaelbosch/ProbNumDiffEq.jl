using ProbNumDiffEq
using Test
using Plots
import ODEProblemLibrary: prob_ode_2Dlinear

prob = remake(prob_ode_2Dlinear, u0=rand(2, 2))

@testset "$alg" for alg in (EK0(), EK1())
    sol = solve(prob, alg)
    @test sol isa ProbNumDiffEq.ProbODESolution

    @test length(sol.u[1]) == length(sol.pu.μ[1])
    @test sol.u[1][:] == sol.pu.μ[1]
    @test sol.u ≈ sol.u_analytic rtol = 1e-4
    @test plot(sol) isa AbstractPlot
end
