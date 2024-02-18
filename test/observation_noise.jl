using Test
using ProbNumDiffEq
using LinearAlgebra, FillArrays
import ODEProblemLibrary: prob_ode_lotkavolterra

prob = prob_ode_lotkavolterra
d = length(prob.u0)
@testset "typeof(R)=$(typeof(R))" for (i, R) in enumerate((
    0.1,
    0.1I,
    0.1Eye(d),
    0.1I(d),
    Diagonal([0.1, 0.2]),
    [0.1 0.01; 0.01 0.1],
    PSDMatrix(0.1 * rand(d, d)),
))
    if i <= 3
        @test_nowarn solve(prob, EK0(pn_observation_noise=R))
    else
        @test_broken solve(prob, EK0(pn_observation_noise=R))
    end
    if i <= 5
        @test_nowarn solve(prob, DiagonalEK1(pn_observation_noise=R))
    else
        @test_broken solve(prob, DiagonalEK1(pn_observation_noise=R))
    end
    @test_nowarn solve(prob, EK1(pn_observation_noise=R))
end
