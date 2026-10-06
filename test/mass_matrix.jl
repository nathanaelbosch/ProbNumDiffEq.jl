using ProbNumDiffEq
using OrdinaryDiffEq
using OrdinaryDiffEqFIRK: RadauIIA5
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

@testset "Non-diagonal and diagonal mass matrices with every solver" begin
    vf(du, u, p, t) = (du .= -u)
    u0 = [1.0, 2.0]
    @testset "$(typeof(M))" for M in ([1.0 0.5; 0.5 1.0], Diagonal([1.0, 2.0]))
        prob = ODEProblem(ODEFunction(vf, mass_matrix=M), u0, (0.0, 1.0))
        @testset "$Alg" for Alg in (EK0, EK1, DiagonalEK1)
            structured = M isa Diagonal && Alg !== EK1
            # Without structure, the `EK0` and the `DiagonalEK1` need to be asked for dense
            kwargs =
                structured || Alg === EK1 ? (;) :
                (covariance_factorization=DenseCovariance,)
            sol = solve(prob, Alg(; kwargs...), abstol=1e-9, reltol=1e-9)
            @test sol.u[end] ≈ exp(-inv(Matrix(M))) * u0 rtol = 1e-8
            structure = structured ? BlockDiagonalCovariance : DenseCovariance
            @test sol.cache.covariance_factorization isa structure
        end
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

    @test_throws "DAE" solve(prob, EK0())
    prob_dense = ODEProblem(
        ODEFunction(rober, mass_matrix=Matrix(M)), prob.u0, prob.tspan, prob.p)
    @test_throws "DAE" solve(prob_dense, EK0(covariance_factorization=DenseCovariance))

    @testset "Initial value with a constraint residual" begin
        prob = remake(prob, u0=[1.0, 0.0, 1e-12])
        ref = solve(prob, RadauIIA5())
        @testset "$Alg, $init" for Alg in (EK1, DiagonalEK1),
            init in (TaylorModeInit(3), SimpleInit())

            sol = solve(prob, Alg(order=3, initialization=init))
            @test sol.retcode == ReturnCode.Success
            @test sol.u[end] ≈ ref.u[end] rtol = 1e-8
        end
    end
end
