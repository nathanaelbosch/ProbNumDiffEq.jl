using ProbNumDiffEq, LinearAlgebra, Test
import ProbNumDiffEq: choose_covariance_structure

K, B, D = IsometricKroneckerCovariance, BlockDiagonalCovariance, DenseCovariance
M = [1.0 0.5; 0.5 1.0]
Mdiag = Diagonal([1.0, 2.0])

@testset "The most structured covariance that all inputs support" begin
    @test choose_covariance_structure(EK0(), I) === K
    @test choose_covariance_structure(EK0(diffusionmodel=DynamicMVDiffusion()), I) === B
    @test choose_covariance_structure(EK0(pn_observation_noise=Mdiag), I) === B
    @test choose_covariance_structure(EK0(), Mdiag) === B
    @test choose_covariance_structure(DiagonalEK1(), I) === B
    @test choose_covariance_structure(EK1(), I) === D
    @test choose_covariance_structure(
        EK1(diffusionmodel=FixedMVDiffusion([1.0, 2.0], false)), M) === D
end

@testset "The EK0 and the DiagonalEK1 use dense covariances only on request" begin
    hint = "covariance_factorization=DenseCovariance"
    @test_throws ["`prior", hint] choose_covariance_structure(EK0(prior=IOUP(3, -1)), I)
    @test_throws ["`mass_matrix", hint] choose_covariance_structure(DiagonalEK1(), M)
    @test_throws ["`pn_observation_noise", hint] choose_covariance_structure(
        EK0(pn_observation_noise=M), I)
    @test choose_covariance_structure(DiagonalEK1(covariance_factorization=D), M) === D
    @test_throws "pass it as a `Diagonal`" choose_covariance_structure(EK0(), [1.0 0; 0 2])
end

@testset "Without a common covariance structure, the error names the inputs" begin
    @test_throws ["`linearization", "`diffusionmodel"] choose_covariance_structure(
        EK1(diffusionmodel=DynamicMVDiffusion()), I)
    @test_throws ["`covariance_factorization", "`diffusionmodel"] choose_covariance_structure(
        EK0(covariance_factorization=K, diffusionmodel=DynamicMVDiffusion()), I)
end

@testset "Printed as the call that constructs it" begin
    @test repr(EK0()) == "EK0()"
    @test repr(DiagonalEK1(order=5, smooth=false)) == "DiagonalEK1(order=5, smooth=false)"
    @test "$(ODEFilter(linearization=FullJacobian(), order=5))" == "EK1(order=5)"
end
