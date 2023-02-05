using ProbNumDiffEq
using LinearAlgebra
using OrdinaryDiffEq
using Random
using Test

A = [-1e-1 -2π; 2π -1e-1]
u0 = [1.0, 1.0]
f(du, u, p, t) = mul!(du, A, u)
tspan = (0.0, 2.0)
prob = ODEProblem(f, u0, tspan)

ref = solve(prob, Vern9(), abstol=1e-9, reltol=1e-9);

sol_iwp = solve(prob, EK1());
err_iwp = norm(ref[end] - sol_iwp[end])
@test err_iwp < 1e-5

A_noisy = A + 1e-3 * randn(MersenneTwister(42), 2, 2)

@testset "Adaptive steps" begin
    sol_ioup_noisy = solve(prob, EK1(prior=IOUP(3, A_noisy)))
    err_ioup_noisy = norm(ref[end] - sol_ioup_noisy[end])
    @test sol_ioup_noisy.destats.nf < sol_iwp.destats.nf
    @test err_ioup_noisy < 2e-5

    sol_ioup = solve(prob, EK1(prior=IOUP(3, A)))
    err_ioup = norm(ref[end] - sol_ioup[end])
    @test sol_ioup.destats.nf < sol_ioup_noisy.destats.nf
    @test err_ioup < 5e-10
end

@testset "Fixed steps" begin
    last_error = 1.0
    for order in (1, 2, 3, 5)
        sol = solve(prob, EK1(
                prior=IOUP(order, A_noisy),
                diffusionmodel=FixedDiffusion(),
            ), adaptive=false, dt=1e-1)
        err = norm(ref[end] - sol[end])
        @test err < last_error
        last_error = err
    end
end

@testset "Different rate types" begin
    @testset "$(typeof(r))" for r in (1, [1, 1], [1 0; 0 1], 1 * I(2))
        sol = solve(prob, EK1(prior=IOUP(3, r)))

        err = norm(ref[end] - sol[end])
        @test err < 5e-6
    end
end
