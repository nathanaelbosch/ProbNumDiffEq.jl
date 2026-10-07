using ProbNumDiffEq, Test
import ProbNumDiffEq as PNDE

@testset "Solvers are printed as the call that constructs them" begin
    @test repr(EK0()) == "EK0()"
    @test repr(DiagonalEK1(order=5, smooth=false)) == "DiagonalEK1(order=5, smooth=false)"
    @test "$(ODEFilter(linearization=FullJacobian(), order=5))" == "EK1(order=5)"
end

@testset "Priors are printed as the call that constructs them" begin
    for prior in (
        IWP(3),
        IWP(dim=2, num_derivatives=3),
        PNDE.remake(IWP(3); elType=Float32),
        IOUP(3, -1),
        IOUP(3, update_rate_parameter=true),
        IOUP(dim=2, num_derivatives=1, rate_parameter=[-1 0; 0 -2]),
        Matern(3, 1.5),
        Matern(dim=2, num_derivatives=3, lengthscale=1),
    )
        rebuilt = include_string(@__MODULE__, repr(prior))
        @test typeof(rebuilt) == typeof(prior)
        @test all(
            f -> isequal(getfield(rebuilt, f), getfield(prior, f)),
            fieldnames(typeof(prior)),
        )
    end
    @test repr(IOUP(3, -1)) == "IOUP(3, -1)"
    @test repr(IOUP(3, update_rate_parameter=true)) == "IOUP(3, update_rate_parameter=true)"
end
