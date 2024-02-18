using ProbNumDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using Test

@testset "Simple UniformScaling mass-matrix" begin
    vf(du, u, p, t) = (du .= u)
    M = -100I
    f = ODEFunction(vf, mass_matrix=M)
    prob = ODEProblem(f, [1.0], (0.0, 1.0))
    ref = solve(prob, RadauIIA5())

    @testset "Correct EK1" begin
        sol = solve(prob, EK1(order=3))
        @test sol.u[end] ≈ ref.u[end] rtol = 1e-10
    end

    @testset "Kronecker working" begin
        N = 10
        prob = ODEProblem(f, ones(N), (0.0, 1.0))
        ek1() = solve(
            prob,
            EK1(smooth=false),
            save_everystep=false,
            dense=false,
            adaptive=false,
            dt=1e-2,
        )
        ek0() = solve(
            prob,
            EK0(smooth=false),
            save_everystep=false,
            dense=false,
            adaptive=false,
            dt=1e-2,
        )
        diagonalek1() = solve(
            prob,
            DiagonalEK1(smooth=false),
            save_everystep=false,
            dense=false,
            adaptive=false,
            dt=1e-2,
        )
        s1 = ek1()
        s0 = ek0()
        s1diag = diagonalek1()

        ref = solve(prob, RadauIIA5(), abstol=1e-9, reltol=1e-6)
        @test s0.u[end] ≈ ref.u[end] rtol = 1e-7
        @test s1.u[end] ≈ ref.u[end] rtol = 1e-7
        @test s1diag.u[end] ≈ ref.u[end] rtol = 1e-7

        @test s1.pu.Σ[1] isa PSDMatrix{<:Number,<:Matrix}
        @test s0.pu.Σ[1] isa PSDMatrix{<:Number,<:ProbNumDiffEq.IsometricKroneckerProduct}

        t1 = @elapsed ek1()
        t0 = @elapsed ek0()
        @test t0 < t1
    end
end

@testset "Robertson in mass-matrix-ODE form" begin
    function rober(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
        du[3] = y₁ + y₂ + y₃ - 1
        return nothing
    end
    M = [
        1 0 0
        0 1 0
        0 0 0
    ]
    M = Diagonal([1, 1, 0])
    f = ODEFunction(rober, mass_matrix=M)
    prob = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4))

    ref = solve(prob, RadauIIA5())
    sol = solve(prob, EK1(order=3))
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-8

    sol = solve(prob, EK1(order=3, initialization=ForwardDiffInit(3)))
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-8

    sol = solve(prob, EK1(order=3, initialization=ClassicSolverInit(RadauIIA5())))
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-8

    sol = solve(prob, EK1(order=3, initialization=SimpleInit()))
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-8

    sol = solve(prob, DiagonalEK1(order=3))
    @test sol.u[end] ≈ ref.u[end] rtol = 1e-8

    @test_throws ArgumentError solve(prob, EK0())
end
