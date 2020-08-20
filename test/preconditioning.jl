using Test
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo


prob = prob_ode_lotkavoltera
prob = remake(prob, tspan=(0.0, 10.0))
prob = ProbNumODE.remake_prob_with_jac(prob)


@testset "Same Solution, different states" begin
    sol1 = solve(prob, EKF0(), q=2, steprule=:constant, dt=0.01, smooth=false,
                 precond_dt=1.0)
    sol2 = solve(prob, EKF0(), q=2, steprule=:constant, dt=0.01, smooth=false)

    # Same solutions
    @test sol1.t ≈ sol2.t
    @test sol1.u.μ ≈ sol2.u.μ
    @test sol1.u.Σ ≈ sol2.u.Σ atol=1e-10

    # Different states
    @test sol1.x != sol2.x

    # Same states if we undo the preconditioning
    PI = sol2.solver.constants.InvPrecond
    @test sol1.x.μ ≈ [PI * x.μ for x in sol2.x]
    @test sol1.x.Σ ≈ [PI * x.Σ * PI' for x in sol2.x]
end
