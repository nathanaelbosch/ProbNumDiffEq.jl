using Test
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo


prob = prob_ode_lotkavoltera
prob = remake(prob, tspan=(0.0, 10.0))
prob = ProbNumODE.remake_prob_with_jac(prob)


@testset "Same Solution, different states" begin
    sol1 = solve(prob, EKF0(), q=2, steprule=:constant, dt=0.1, smooth=false,
                 precond_dt=1.0)
    sol2 = solve(prob, EKF0(), q=2, steprule=:constant, dt=0.1, smooth=false)

    # Same solutions
    @test sol1.t ≈ sol2.t
    @test sol1.pu.μ ≈ sol2.pu.μ
    @test sol1.pu.Σ ≈ sol2.pu.Σ atol=1e-10

    # Different states
    @test sol1.x != sol2.x

    # Same states if we undo the preconditioning
    PI = sol2.solver.constants.InvPrecond
    @test sol1.x.μ ≈ [PI * x.μ for x in sol2.x]
    @test sol1.x.Σ ≈ [PI * x.Σ * PI' for x in sol2.x]
end


@testset "Condition numbers of A,Q" begin
    h = 0.1*rand()
    σ = rand()

    d, q = 2, 3

    A!, Q! = ProbNumODE.ibm(d, q; precond_dt=1.0)
    Ah = diagm(0 => ones(d*(q+1)))
    Qh = zeros(d*(q+1), d*(q+1))
    A!(Ah, h)
    Q!(Qh, h, σ^2)

    A!_p, Q!_p = ProbNumODE.ibm(d, q; precond_dt=h)
    Ah_p = diagm(0 => ones(d*(q+1)))
    Qh_p = zeros(d*(q+1), d*(q+1))
    A!_p(Ah_p, h)
    Q!_p(Qh_p, h, σ^2)

    @info "Condition numbers" cond(Qh) cond(Qh_p) cond(Ah) cond(Ah_p)

    @test cond(Qh) > cond(Qh_p)
    @test cond(Qh) > cond(Qh_p)^2
end


@testset "Compare internals during solve" begin
    integ1 = init(prob, EKF1(), q=2, steprule=:constant, dt=0.1, smooth=false,
                  precond_dt=1.0)
    integ2 = init(prob, EKF1(), q=2, steprule=:constant, dt=0.1, smooth=false)
    PI = integ2.constants.InvPrecond

    for i in 1:20
        if i>1
            step!(integ1)
            step!(integ2)
        end

        @test !(integ1.cache.x ≈ integ2.cache.x)
        @test integ1.cache.x ≈ PI * integ2.cache.x

        @test integ1.cache.du ≈ integ2.cache.du
        @test integ1.cache.ddu ≈ integ2.cache.ddu

        @test integ1.constants.E0 * integ1.cache.x ≈ integ2.constants.E0 * integ2.cache.x
        @test integ1.constants.E1 * integ1.cache.x ≈ integ2.constants.E1 * integ2.cache.x
        @test integ1.cache.h ≈ integ2.cache.h
        @test integ1.cache.H * PI ≈ integ2.cache.H
        i > 1 && @test_broken integ1.cache.H * integ1.cache.x ≈ integ2.cache.H * integ2.cache.x
        @test integ1.cache.measurement ≈ integ2.cache.measurement
    end
end
