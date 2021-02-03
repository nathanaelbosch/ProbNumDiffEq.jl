using ODEFilters
using Test
using Sundials

using DiffEqProblemLibrary.DAEProblemLibrary: importdaeproblems; importdaeproblems()
import DiffEqProblemLibrary.DAEProblemLibrary: prob_dae_resrob


@testset "DAE Solver" begin
    prob = prob_dae_resrob
    appxsol = solve(prob, IDA())
    sol = solve(prob, DAE_EK1(order=4), dense=true, alias_du0=true)
    @test sol isa ODEFilters.ProbODESolution
    @test sol[end] ≈ appxsol[end] rtol=1e-4
end
