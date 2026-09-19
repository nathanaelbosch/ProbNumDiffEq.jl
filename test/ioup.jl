using ProbNumDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test
import ODEProblemLibrary: prob_ode_lotkavolterra

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

@testset "MBF backward pass leaves the IOUP rate parameter at its forward-final value" begin
    # NOTE: `sol.cache` (a.k.a. `sol.interp.cache`) is a *separate* cache object built by
    # `SciMLBase.build_solution`, never mutated during stepping/smoothing -- it always
    # keeps its zero-initialized `rate_parameter`. The cache that `_smooth_solution_mbf!`
    # actually mutates is `integ.cache`, so this checks that one directly via `init`/
    # `solve!` instead of going through the high-level `solve` / `sol.cache`.
    lvprob = prob_ode_lotkavolterra
    integ_s = OrdinaryDiffEq.init(
        lvprob, EK1(prior=IOUP(3, update_rate_parameter=true), smooth=true); dense=false,
    )
    OrdinaryDiffEq.solve!(integ_s)
    integ_f = OrdinaryDiffEq.init(
        lvprob, EK1(prior=IOUP(3, update_rate_parameter=true), smooth=false);
        dense=false)
    OrdinaryDiffEq.solve!(integ_f)
    @test integ_s.cache.prior.rate_parameter ≈ integ_f.cache.prior.rate_parameter
end
