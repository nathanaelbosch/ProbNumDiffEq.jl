using ProbNumDiffEq
using LinearAlgebra
using Random
using Test

@testset "Linear ODE" begin
    f!(du, u, p, t) = @. du = p * u
    u0 = [1.0]
    tspan = (0.0, 10.0)
    p = -1
    prob = ODEProblem(f!, u0, tspan, p)

    u(t) = u0 * exp(p * t)
    uend = u(tspan[2])

    sol0 = solve(prob, EK0(order=3))
    sol1 = solve(prob, EK1(order=3))
    solexp = solve(prob, ExpEK(L=p, order=3))
    solros = solve(prob, RosenbrockExpEK(order=3))

    err0 = norm(uend - sol0.u[end])
    @test err0 < 1e-7
    err1 = norm(uend - sol1.u[end])
    @test err1 < 1e-9
    errexp = norm(uend - solexp.u[end])
    @test errexp < 1e-10
    errros = norm(uend - solros.u[end])
    @test errros < 1e-13

    @test errros < errexp < err1 < err0

    @test length(solexp) < length(sol0)
    @test solexp.stats.nf < sol0.stats.nf
    @test length(solexp) < length(sol1)
    @test solexp.stats.nf < sol1.stats.nf

    @test length(solros) < length(sol0)
    @test solros.stats.nf < sol0.stats.nf
    @test length(solros) < length(sol1)
    @test solros.stats.nf < sol1.stats.nf
end
