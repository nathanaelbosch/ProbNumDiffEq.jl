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

@testset "$S, adaptive=$adaptive" for S in (EK0, EK1, DiagonalEK1),
    adaptive in (true, false)

    sol = solve(prob, S(linearize_at=Posterior()); adaptive, dt=0.05)
    @test sol.retcode == ReturnCode.Success
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-3
    @test sol.stats.naccept < sol.stats.nnonliniter < 3sol.stats.naccept
    @test sol.stats.nnonlinconvfail == 0
end

@testset "The iteration stabilizes the EK0" begin
    @test solve(prob, EK0(), adaptive=false, dt=0.05).retcode != ReturnCode.Success
    sol = solve(prob, EK0(linearize_at=Posterior()), adaptive=false, dt=0.05)
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-3
end

@testset "On a linear ODE, $S gives the same result after one more linearization" for (
    S,
    A,
) in
                                                                                      (
    EK1 => [-1.0 2.0; -2.0 -1.0],
    DiagonalEK1 => Diagonal([-1.0, -2.0]),
)
    linear = ODEProblem((du, u, p, t) -> mul!(du, A, u), [1.0, 0.5], (0.0, 3.0))
    sol = solve(linear, S(), adaptive=false, dt=0.1)
    sol_post = solve(linear, S(linearize_at=Posterior()), adaptive=false, dt=0.1)
    @test sol_post.stats.nnonliniter == 2sol_post.stats.naccept
    @test [x.μ for x in sol_post.pu] ≈ [x.μ for x in sol.pu]
    @test [Matrix(x.Σ) for x in sol_post.pu] ≈ [Matrix(x.Σ) for x in sol.pu]
    @test sol_post.pnstats.log_likelihood ≈ sol.pnstats.log_likelihood
end

@testset "Steps whose iteration doesn't converge" begin
    alg = EK1(linearize_at=Posterior(maxiters=1))
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

@test_throws ArgumentError Posterior(maxiters=0)
