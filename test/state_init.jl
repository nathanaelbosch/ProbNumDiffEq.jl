# Goal: Test the correctness of state initialization
using ProbNumDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Test

import ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_pleiades

@testset "Testproblem" begin
    d = 2
    q = 6
    D = d * (q + 1)

    a, b = 1.1, -0.5
    f(u, p, t) = [a * u[1], b * u[2]]
    u0 = [0.1, 1.0]
    tspan = (0.0, 5.0)
    t0, T = tspan
    prob = ODEProblem(f, u0, tspan)
    p = prob.p

    # True Solutions and derivatives
    u(t) = [a^0 * u0[1] * exp(a * t), u0[2] * exp(b * t)]
    du(t) = [a^1 * u0[1] * exp(a * t), b * u0[2] * exp(b * t)]
    ddu(t) = [a^2 * u0[1] * exp(a * t), (b)^2 * u0[2] * exp(b * t)]
    dddu(t) = [a^3 * u0[1] * exp(a * t), (b)^3 * u0[2] * exp(b * t)]
    ddddu(t) = [a^4 * u0[1] * exp(a * t), (b)^4 * u0[2] * exp(b * t)]
    dddddu(t) = [a^5 * u0[1] * exp(a * t), (b)^5 * u0[2] * exp(b * t)]
    ddddddu(t) = [a^6 * u0[1] * exp(a * t), (b)^6 * u0[2] * exp(b * t)]
    true_init_states =
        [u(t0); du(t0); ddu(t0); dddu(t0); ddddu(t0); dddddu(t0); ddddddu(t0)]

    f!(du, u, p, t) = (du .= f(u, p, t))
    prob = ODEProblem{true,true}(f!, u0, tspan)

    @testset "`taylormode_get_derivatives`" begin
        dfs = ProbNumDiffEq.taylormode_get_derivatives(
            prob.u0,
            prob.f,
            prob.p,
            prob.tspan[1],
            q,
        )
        @test length(dfs) == q + 1
        @test true_init_states ≈ vcat(dfs...)
    end

    @testset "Taylormode: `initial_update!`" begin
        integ = init(prob, EK0(order=q))
        ProbNumDiffEq.initial_update!(integ, integ.cache, TaylorModeInit())
        x = integ.cache.x
        @test reshape(x.μ, :, 2)'[:] ≈ true_init_states
    end

    @testset "Low-order exact init via ClassiSolverInit: `initial_update!`" begin
        @test_nowarn init(
            prob,
            EK0(order=1, initialization=ClassicSolverInit(init_on_ddu=true)),
        )
        @test_nowarn init(
            prob,
            EK0(order=2, initialization=ClassicSolverInit(init_on_ddu=false)),
        )
        @test_broken init(
            prob,
            EK0(order=2, initialization=ClassicSolverInit(init_on_ddu=true)),
        )

        @test_nowarn init(
            prob,
            EK1(order=1, initialization=ClassicSolverInit(init_on_ddu=true)),
        )
        @test_nowarn init(
            prob,
            EK1(order=2, initialization=ClassicSolverInit(init_on_ddu=true)),
        )
        integ =
            init(prob, EK1(order=2, initialization=ClassicSolverInit(init_on_ddu=true)))
        ProbNumDiffEq.initial_update!(integ, integ.cache, integ.alg.initialization)
        x = integ.cache.x
        @test reshape(x.μ, :, 2)'[:] ≈ true_init_states[1:(2+1)*d]
    end
end

@testset "Compare TaylorModeInit and ClassicSolverInit" begin
    prob = prob_ode_fitzhughnagumo
    d = length(prob.u0)

    integ1 = init(prob, EK0(order=8))
    tm_init = integ1.cache.x.μ
    Proj1 = integ1.cache.Proj

    @testset "Order $o" for o in (1, 2, 3, 4, 5)
        integ2 =
            init(prob, EK1(order=o, initialization=ClassicSolverInit(init_on_ddu=true)))
        rk_init = integ2.cache.x.μ
        Proj2 = integ2.cache.Proj

        # These are fit via the initial values + autodiff, and should be good
        @test Proj1(0) * tm_init ≈ Proj2(0) * rk_init
        @test Proj1(1) * tm_init ≈ Proj2(1) * rk_init
        (o > 1) && @test Proj1(2) * tm_init ≈ Proj2(2) * rk_init

        # # Test if the others are correct, up to order 5
        (5 > o > 2) && @test Proj1(3) * tm_init ≈ Proj2(3) * rk_init rtol = 1e-1
        (5 > o > 3) && @test Proj1(4) * tm_init ≈ Proj2(4) * rk_init rtol = 1e-1
        # (o == 5) && @test Proj1(3) * tm_init ≈ Proj2(3) * rk_init rtol=6e-1
        # (o == 5) && @test Proj1(4) * tm_init ≈ Proj2(4) * rk_init rtol=2e-2
        # (o == 5) && @test Proj1(5) * tm_init ≈ Proj2(5) * rk_init rtol=8e-1

        # Test if the covariance covers the true error
        for i in 3:o
            _rk = Proj2(i) * rk_init
            _tm = Proj1(i) * tm_init
            err = _rk .- _tm
            C = ProbNumDiffEq.X_A_Xt(integ2.cache.x.Σ, Proj2(i))
            Cm = Matrix(C)
            @assert isdiag(Cm)
            whitened_err = err ./ sqrt.(diag(Cm))
            @test all(abs.(whitened_err) .< 4e-1)
        end
    end
end
