# Goal: Make sure some combinations that raise errors do so
using Test
using ProbNumDiffEq
import ODEProblemLibrary: prob_ode_lotkavolterra

@testset "Fixed-timestep requires dt" begin
    prob = prob_ode_lotkavolterra
    @test_throws ErrorException solve(prob, EK0(), adaptive=false)
    @test_nowarn solve(
        prob,
        EK0(smooth=false),
        save_everystep=false,
        adaptive=false,
        dt=0.1,
    )
end

@testset "`dense=true` requires `smooth=true`" begin
    prob = prob_ode_lotkavolterra
    @test_throws ErrorException solve(prob, EK0(smooth=false), dense=true)
end

@testset "`save_everystep=false` requires `smooth=false`" begin
    prob = prob_ode_lotkavolterra
    @test_throws ErrorException solve(prob, EK0(smooth=true), save_everystep=false)
end

@testset "Invalid prior" begin
    prob = prob_ode_lotkavolterra
    @test_throws DimensionMismatch solve(prob, EK0(prior=IWP(dim=3, num_derivatives=2)))
    prior = IOUP(num_derivatives=1, rate_parameter=3, update_rate_parameter=true)
    @test_throws ArgumentError solve(prob, EK0(; prior))
end
