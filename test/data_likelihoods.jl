using ProbNumDiffEq, Plots, Statistics, LinearAlgebra
import ProbNumDiffEq as PNDE
import ODEProblemLibrary: prob_ode_lotkavolterra
using FillArrays
using Test

prob = prob_ode_lotkavolterra
sol = solve(prob, EK1(diffusionmodel=FixedDiffusion()), abstol=1e-9, reltol=1e-9);
σ = 5e-1
times = range(prob.tspan..., length=3)[2:end];
obss = [mean(sol(t)) + σ * randn(length(prob.u0)) for t in times];
data = (t=times, u=obss);

DT = 2e-2

function compare_data_likelihoods(alg; kwargs...)
    dalton_ll = @test_nowarn PNDE.dalton_data_loglik(
        prob, remake(alg, smooth=false); kwargs...)
    filtering_ll = @test_nowarn PNDE.filtering_data_loglik(
        prob, remake(alg, smooth=false); kwargs...)
    fenrir_ll = PNDE.fenrir_data_loglik(
        prob, remake(alg, smooth=true); kwargs...,
    )
    @test dalton_ll ≈ filtering_ll rtol = 1e-6
    @test dalton_ll ≈ fenrir_ll rtol = 1e-6
end

kwargs = (
    observation_noise_cov=σ^2,
    data=data,
    adaptive=false, dt=DT,
    dense=false,
)
@testset "Compare data likelihoods" begin
    @testset "$alg" for alg in (
        # EK0
        EK0(),
        EK0(diffusionmodel=FixedDiffusion()),
        EK0(prior=IOUP(3, -1)),
        EK0(prior=Matern(3, 1.5)),
        # EK1
        EK1(),
        EK1(diffusionmodel=FixedDiffusion()),
        # EK1(diffusionmodel=FixedMVDiffusion(rand(2), false)), # not yet supported
        EK1(prior=IOUP(3, -1)),
        EK1(prior=Matern(3, 1.5)),
        EK1(prior=IOUP(3, update_rate_parameter=true)),
        # DiagonalEK1
        DiagonalEK1(),
        DiagonalEK1(diffusionmodel=FixedDiffusion()),
        DiagonalEK1(diffusionmodel=FixedMVDiffusion(rand(2), false)),
    )
        compare_data_likelihoods(alg; kwargs...)
    end
end

@testset "Partial observations" begin
    H = [1 0;]
    data_part = (t=times, u=[H * d for d in obss])
    compare_data_likelihoods(
        EK1();
        observation_matrix=H,
        observation_noise_cov=σ^2,
        data=data_part,
        adaptive=false, dt=DT,
        dense=false,
    )
    @test_broken compare_data_likelihoods(
        DiagonalEK1();
        observation_matrix=H,
        observation_noise_cov=σ^2,
        data=data_part,
        adaptive=false, dt=DT,
        dense=false,
    )
end

@testset "Observation noise types: $(typeof(Σ))" for Σ in (
    σ^2,
    σ^2 * I,
    σ^2 * I(2),
    σ^2 * Eye(2),
    Diagonal([σ^2 0; 0 2σ^2]),
    [σ^2 0; 0 2σ^2],
    (A = randn(2, 2); A'A),
    (PSDMatrix(randn(2, 2))),
)
    @testset "$alg" for alg in (EK0(), DiagonalEK1(), EK1())
        if alg isa EK0 && !(
            Σ isa Number || Σ isa UniformScaling ||
            Σ isa Diagonal{<:Number,<:FillArrays.Fill}
        )
            continue
        end
        if alg isa DiagonalEK1 && !(Σ isa Number || Σ isa UniformScaling || Σ isa Diagonal)
            continue
        end
        compare_data_likelihoods(
            alg;
            observation_noise_cov=Σ,
            data=data,
            adaptive=false, dt=DT,
            dense=false,
        )
    end
end
