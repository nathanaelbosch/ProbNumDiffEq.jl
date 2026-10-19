using ProbNumDiffEq
using Test
using LinearAlgebra
using OrdinaryDiffEq: Tsit5

function fhn!(du, u, p, t)
    du[1] = p * (u[1] - u[1]^3 / 3 + u[2])
    du[2] = -(u[1] - 0.2 + 0.2u[2]) / p
end
prob = ODEProblem(fhn!, [-1.0, 1.0], (0.0, 5.0), 3.0)
ref = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)

@testset "$L, adaptive=$adaptive" for L in
                                      (ZeroJacobian(), FullJacobian(), DiagonalJacobian()),
    adaptive in (true, false)

    sol = solve(prob, ODEFilter(linearization=Iterated(L)); adaptive, dt=0.05)
    @test sol.retcode == ReturnCode.Success
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-3
    @test sol.stats.naccept < sol.stats.nnonliniter < 3sol.stats.naccept
    @test sol.stats.nnonlinconvfail == 0
end

@testset "The iteration stabilizes the EK0" begin
    @test solve(prob, EK0(), adaptive=false, dt=0.05).retcode != ReturnCode.Success
    alg = ODEFilter(linearization=Iterated(ZeroJacobian()))
    @test solve(prob, alg, adaptive=false, dt=0.05).u[end] ≈ ref.u[end] rtol = 1e-3
end

linear_cases = (
    FullJacobian() => [-1.0 2.0; -2.0 -1.0],
    DiagonalJacobian() => Diagonal([-1.0, -2.0]),
)
# On a linear ODE, the second linearization agrees with the first
@testset "Linear ODE, $L" for (L, A) in linear_cases
    linear = ODEProblem((du, u, p, t) -> mul!(du, A, u), [1.0, 0.5], (0.0, 3.0))
    sol = solve(linear, ODEFilter(linearization=L), adaptive=false, dt=0.1)
    sol_iter = solve(linear, ODEFilter(linearization=Iterated(L)), adaptive=false, dt=0.1)
    @test sol_iter.stats.nnonliniter == 2sol_iter.stats.naccept
    @test [x.μ for x in sol_iter.pu] ≈ [x.μ for x in sol.pu]
    @test [Matrix(x.Σ) for x in sol_iter.pu] ≈ [Matrix(x.Σ) for x in sol.pu]
    @test sol_iter.pnstats.log_likelihood ≈ sol.pnstats.log_likelihood
end

@testset "Steps whose iteration doesn't converge" begin
    alg = ODEFilter(linearization=Iterated(FullJacobian(), maxiters=1))
    # are rejected with adaptive steps, so that every accepted step converged
    sol = solve(prob, alg)
    @test sol.retcode == ReturnCode.Success
    @test sol.stats.nnonlinconvfail > 0
    @test sol.stats.nnonliniter == sol.stats.naccept + sol.stats.nnonlinconvfail
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-3
    # and accepted with fixed steps
    sol = solve(prob, alg, adaptive=false, dt=0.05)
    @test sol.stats.nnonlinconvfail > 0
    @test sol.stats.naccept == 100
end

@test_throws ArgumentError Iterated(FullJacobian(), maxiters=0)
@test_throws ArgumentError Iterated(FullJacobian(), tol=0)
