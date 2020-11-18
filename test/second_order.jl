using ODEFilters
using Test


function vanderpol!(ddu, du, u, p, t)
    μ = p[1]
    @. ddu = μ * ((1-u^2) * du - u)
end
du0 = [0.0]
u0 = [2.0]
tspan = (0.0, 6.3)
p = [1e1]
prob = SecondOrderODEProblem(vanderpol!, du0, u0, tspan, p)


@testset "Solve with EKF0" begin
    @test begin
        sol1 = solve(prob, EKF0(order=3))
        sol1 isa ODEFilters.ProbODESolution
    end
end


@testset "Solve with SecondOrderEKF0" begin
    @test begin
        sol2 = solve(prob, SecondOrderEKF0(order=3))
        sol2 isa ODEFilters.ProbODESolution
    end
end


@testset "EKF0 and 2ndEKF0 states" begin
    integ1 = init(prob, EKF0(order=3), abstol=1e-8, reltol=1e-8)
    integ2 = init(prob, SecondOrderEKF0(order=3), abstol=1e-8, reltol=1e-8)

    @test integ1.u == integ2.u

    @test integ2.cache.x.μ[1] == integ1.cache.x.μ[2]
    @test integ2.cache.x.μ[2] == integ1.cache.x.μ[1]
    @test integ2.cache.x.μ[end] == integ1.cache.x.μ[end-1]

    @test length(integ1.cache.x.μ) == 2*(3+1)
    @test length(integ2.cache.x.μ) == 1*(3+2)

    sol1 = solve!(integ1)
    sol2 = solve!(integ2)
    @test integ1.u ≈ integ2.u
    @test sol1.u[end] ≈ sol1.u[end]
end
