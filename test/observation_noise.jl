using Test
using ProbNumDiffEq
using LinearAlgebra, FillArrays
import ODEProblemLibrary: prob_ode_lotkavolterra

prob = prob_ode_lotkavolterra
d = length(prob.u0)
# Isotropic noise fits all covariance structures, diagonal noise block-diagonal and dense
# ones, and other noise only dense ones, which the `EK0` and the `DiagonalEK1` use only
# on request
@testset "typeof(R)=$(typeof(R))" for (i, R) in enumerate((
    0.1,
    0.1I,
    0.1Eye(d),
    0.1I(d),
    Diagonal([0.1, 0.2]),
    [0.1 0.01; 0.01 0.1],
    PSDMatrix(0.1 * rand(d, d)),
))
    @testset "$Alg" for Alg in (EK0, DiagonalEK1, EK1)
        if i > 5 && Alg !== EK1
            @test_throws ArgumentError solve(prob, Alg(pn_observation_noise=R))
            alg = Alg(pn_observation_noise=R, covariance_factorization=DenseCovariance)
            @test solve(prob, alg).retcode == ReturnCode.Success
            continue
        end
        sol = @test_nowarn solve(prob, Alg(pn_observation_noise=R))
        default = (EK0=IsometricKroneckerCovariance, DiagonalEK1=BlockDiagonalCovariance,
            EK1=DenseCovariance)[Symbol(Alg)]
        structure = i <= 3 || Alg === EK1 ? default : BlockDiagonalCovariance
        @test sol.cache.covariance_factorization isa structure
    end
end
