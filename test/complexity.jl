#=
Test that the Kronecker EK0 scales linearly with the ODE dimension, not cubically as the EK1
=#
using ProbNumDiffEq, LinearAlgebra
import LinearRegression: linregress, slope
using Test, SafeTestsets

@testset "Scaling with ODE dimension" begin
    f(du, u, p, t) = mul!(du, -0.9I, u)
    tspan = (0.0, 1.0)
    prob = ODEProblem(f, ones(1), tspan)

    NUMRUNS = 20

    @testset "Order 1 + perfect init + no smoothing" begin
        time_dim(d; Alg) = begin
            _prob = remake(prob, u0=ones(d))
            tmin = Inf
            for _ in 1:NUMRUNS
                integ = init(_prob,
                    Alg(
                        smooth=false,
                        order=1,
                        initialization=ClassicSolverInit(),
                    ),
                    dense=false, save_everystep=false,
                    adaptive=false, dt=1e-2,
                )
                t = @elapsed solve!(integ)
                tmin = min(tmin, t)
            end
            return tmin
        end

        dims_ek0 = 2 .^ (8:15)
        times_ek0 = [time_dim(d; Alg=EK0) for d in dims_ek0]
        dims_ek1 = 2 .^ (2:6)
        times_ek1 = [time_dim(d; Alg=EK1) for d in dims_ek1]

        lr_ek0 = linregress(log.(dims_ek0), log.(times_ek0))
        @test slope(lr_ek0)[1] ≈ 1 atol = 0.1

        lr_ek1 = linregress(log.(dims_ek1), log.(times_ek1))
        @test slope(lr_ek1)[1] ≈ 2 atol = 0.2
        # This is what we would actually expect, not sure what's going wrong:
        @test_broken slope(lr_ek1)[1] ≈ 3 atol = 0.1
    end

    @testset "Order 3 + Taylor-init + no smoothing" begin
        time_dim(d; Alg) = begin
            _prob = remake(prob, u0=ones(d))
            tmin = Inf
            for _ in 1:NUMRUNS
                integ = init(_prob, Alg(smooth=false),
                    dense=false, save_everystep=false,
                    adaptive=false, dt=1e-2)
                t = @elapsed solve!(integ)
                tmin = min(tmin, t)
            end
            return tmin
        end

        dims_ek0 = 2 .^ (8:15)
        times_ek0 = [time_dim(d; Alg=EK0) for d in dims_ek0]
        dims_ek1 = 2 .^ (2:5)
        times_ek1 = [time_dim(d; Alg=EK1) for d in dims_ek1]

        lr_ek0 = linregress(log.(dims_ek0), log.(times_ek0))
        @test slope(lr_ek0)[1] ≈ 1 atol = 0.1

        lr_ek1 = linregress(log.(dims_ek1), log.(times_ek1))
        @test slope(lr_ek1)[1] ≈ 2 atol = 0.5
        # This is what we would actually expect, not sure what's going wrong:
        @test_broken slope(lr_ek1)[1] ≈ 3 atol = 0.1
    end

    @testset "Order 3 with smoothing and everyting" begin
        time_dim(d; Alg) = begin
            _prob = remake(prob, u0=ones(d))
            tmin = Inf
            for _ in 1:NUMRUNS
                integ = init(_prob, Alg(), adaptive=false, dt=1e-2)
                t = @elapsed solve!(integ)
                tmin = min(tmin, t)
            end
            return tmin
        end

        dims_ek0 = 2 .^ (8:13)
        times_ek0 = [time_dim(d; Alg=EK0) for d in dims_ek0]
        dims_ek1 = 2 .^ (1:4)
        times_ek1 = [time_dim(d; Alg=EK1) for d in dims_ek1]

        lr_ek0 = linregress(log.(dims_ek0), log.(times_ek0))
        @test slope(lr_ek0)[1] ≈ 1 atol = 0.3

        lr_ek1 = linregress(log.(dims_ek1), log.(times_ek1))
        @test slope(lr_ek1)[1] ≈ 2 atol = 0.5
        # This is what we would actually expect, not sure what's going wrong:
        @test_broken slope(lr_ek1)[1] ≈ 3 atol = 0.1
    end
end
