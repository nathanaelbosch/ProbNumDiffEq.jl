using ProbNumDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

α = 1e-1
A = [-α -2π; 2π -α]
u0 = [1.0, 1.0]
f(du, u, p, t) = mul!(du, A, u)
tspan = (0.0, 2.0)
prob = ODEProblem(f, u0, tspan)

ref = solve(prob, Vern9(), abstol=1e-9, reltol=1e-9);

sol_iwp = solve(prob, EK1());
err_iwp = norm(ref.u[end] - sol_iwp.u[end])
@test err_iwp < 1e-5

A_noisy = A + 1e-3 * randn(MersenneTwister(42), 2, 2)

@testset "Adaptive steps" begin
    sol_ioup_noisy = solve(prob, EK1(prior=IOUP(3, A_noisy)))
    err_ioup_noisy = norm(ref.u[end] - sol_ioup_noisy.u[end])
    @test sol_ioup_noisy.stats.nf < sol_iwp.stats.nf
    @test err_ioup_noisy < 2e-5

    sol_ioup = solve(prob, EK1(prior=IOUP(3, A)))
    err_ioup = norm(ref.u[end] - sol_ioup.u[end])
    @test sol_ioup.stats.nf < sol_ioup_noisy.stats.nf
    @test err_ioup < 5e-10
end

@testset "Fixed steps" begin
    last_error = 1.0
    for order in (1, 2, 3, 5)
        sol = solve(prob, EK1(
            prior=IOUP(order, A_noisy),
            diffusionmodel=FixedDiffusion(),
        ), adaptive=false, dt=1e-1)
        err = norm(ref.u[end] - sol.u[end])
        @test err < last_error
        last_error = err
    end
end

@testset "Different rate types" begin
    @testset "$(typeof(r))" for r in (-α, [-α, -α], [-α 0; 0 -α], -α * I(2))
        sol = solve(prob, EK1(prior=IOUP(3, r)))

        err = norm(ref.u[end] - sol.u[end])
        @test err < 6e-6
    end
end

@testset "Transition matrices are reused only for an unchanged step size" begin
    # tspan is not a multiple of dt, so the last step is shorter and must recompute
    _prob = ODEProblem(f, u0, (0.0, 0.95))
    for alg in (EK1(order=3, smooth=false), EK1(prior=IOUP(3, -1.0), smooth=false))
        integ = init(_prob, alg; adaptive=false, dt=0.1, dense=false)
        forced = init(_prob, alg; adaptive=false, dt=0.1, dense=false)
        while integ.t < _prob.tspan[2]
            forced.cache.dt_last = NaN  # NaN never equals dt, so this always recomputes
            step!(forced)
            step!(integ)
            @test integ.cache.dt_last == integ.dt
            @test integ.cache.x.μ == forced.cache.x.μ
            @test Matrix(integ.cache.x.Σ) == Matrix(forced.cache.x.Σ)
        end
        @test integ.t == forced.t == _prob.tspan[2]
    end
end
