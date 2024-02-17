#=
Test that the Kronecker EK0 scales linearly with the ODE dimension, not cubically as the EK1
=#
using ProbNumDiffEq, LinearAlgebra
import LinearRegression: linregress, slope
using Test, SafeTestsets

@testset "Scaling with ODE dimension" begin
    f(du, u, p, t) = mul!(du, -0.9I, u)
    jac(J, u, p, t) = @simd ivdep for i in 1:size(J, 1)
        J[i, i] = -0.9
    end
    tspan = (0.0, 1.0)
    prob = ODEProblem(f, ones(1), tspan)

    NUMRUNS = 20

    _timer(d, alg; kwargs...) = begin
        _prob = remake(
            prob,
            u0=ones(d),
            f=ODEFunction(f; jac=jac, jac_prototype=Diagonal(ones(d))),
        )
        tmin = Inf
        for _ in 1:NUMRUNS
            integ = init(_prob, alg; adaptive=false, dt=1e-2, kwargs...)
            t = @elapsed solve!(integ)
            tmin = min(tmin, t)
        end
        return tmin
    end

    @testset "Order 1 + perfect init + no smoothing" begin
        t(d, Alg) = _timer(
            d, Alg(smooth=false, order=1, initialization=ClassicSolverInit());
            dense=false, save_everystep=false,
        )

        dims_ek0 = 2 .^ (8:15)
        times_ek0 = [t(d, EK0) for d in dims_ek0]
        lr_ek0 = linregress(log.(dims_ek0), log.(times_ek0))
        @test slope(lr_ek0)[1] ≈ 1 atol = 1

        dims_ek1 = 2 .^ (3:6)
        times_ek1 = [t(d, EK1) for d in dims_ek1]
        lr_ek1 = linregress(log.(dims_ek1), log.(times_ek1))
        @test slope(lr_ek1)[1] ≈ 3 atol = 1

        dims_dek1 = 2 .^ (4:10)
        times_dek1 = [t(d, DiagonalEK1) for d in dims_dek1]
        lr_dek1 = linregress(log.(dims_dek1), log.(times_dek1))
        @test slope(lr_dek1)[1] ≈ 1 atol = 1
    end

    @testset "Order 3 + Taylor-init + no smoothing" begin
        t(d, Alg) = _timer(d, Alg(smooth=false); dense=false, save_everystep=false)

        dims_ek0 = 2 .^ (8:15)
        times_ek0 = [t(d, EK0) for d in dims_ek0]
        lr_ek0 = linregress(log.(dims_ek0), log.(times_ek0))
        @test slope(lr_ek0)[1] ≈ 1 atol = 1

        dims_ek1 = 2 .^ (3:6)
        times_ek1 = [t(d, EK1) for d in dims_ek1]
        lr_ek1 = linregress(log.(dims_ek1), log.(times_ek1))
        @test slope(lr_ek1)[1] ≈ 3 atol = 1

        dims_dek1 = 2 .^ (4:10)
        times_dek1 = [t(d, DiagonalEK1) for d in dims_dek1]
        lr_dek1 = linregress(log.(dims_dek1), log.(times_dek1))
        @test slope(lr_dek1)[1] ≈ 1 atol = 1
    end

    @testset "Order 3 with smoothing and everyting" begin
        t(d, Alg) = _timer(d, Alg())

        dims_ek0 = 2 .^ (8:13)
        times_ek0 = [t(d, EK0) for d in dims_ek0]
        lr_ek0 = linregress(log.(dims_ek0), log.(times_ek0))
        @test slope(lr_ek0)[1] ≈ 1 atol = 1

        dims_ek1 = 2 .^ (3:6)
        times_ek1 = [t(d, EK1) for d in dims_ek1]
        lr_ek1 = linregress(log.(dims_ek1), log.(times_ek1))
        @test slope(lr_ek1)[1] ≈ 3 atol = 1

        dims_dek1 = 2 .^ (4:10)
        times_dek1 = [t(d, DiagonalEK1) for d in dims_dek1]
        lr_dek1 = linregress(log.(dims_dek1), log.(times_dek1))
        @test slope(lr_dek1)[1] ≈ 1 atol = 1
    end
end
