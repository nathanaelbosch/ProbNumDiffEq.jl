using ProbNumDiffEq, Plots, Statistics, LinearAlgebra
import ProbNumDiffEq as PNDE
import ODEProblemLibrary: prob_ode_lotkavolterra
using Test

prob = prob_ode_lotkavolterra
sol = solve(prob, EK1(diffusionmodel=FixedDiffusion()), abstol=1e-9, reltol=1e-9);
σ = 5e-1
times = range(prob.tspan..., length=3)[2:end];
obss = [mean(sol(t)) + σ * randn(length(prob.u0)) for t in times];
data = (t=times, u=obss);

DT = 2e-2

@testset "Compare data likelihoods" begin
    dalton_ll = @test_nowarn PNDE.dalton_data_loglik(
        prob,
        EK1(smooth=false, diffusionmodel=FixedDiffusion());
        # observation_matrix=I,
        observation_noise_cov=σ^2,
        data,
        adaptive=false, dt=DT,
        dense=false,
    )

    filtering_ll = @test_nowarn PNDE.filtering_data_loglik(
        prob,
        EK1(smooth=false, diffusionmodel=FixedDiffusion());
        # observation_matrix=I,
        observation_noise_cov=σ^2,
        data,
        adaptive=false, dt=DT,
        dense=false,
    )

    fenrir_ll = @test_nowarn PNDE.fenrir_data_loglik(
        prob,
        EK1(smooth=true, diffusionmodel=FixedDiffusion());
        # observation_matrix=I,
        observation_noise_cov=σ^2,
        data,
        adaptive=false, dt=DT,
        dense=false,
    )

    @test dalton_ll ≈ filtering_ll
    @test dalton_ll ≈ fenrir_ll
end

@testset "EK0 is not (yet) supported" begin
    for ll in (PNDE.dalton_data_loglik, PNDE.filtering_data_loglik)
        @test_broken ll(prob, EK0(smooth=false);
                        observation_noise_cov=σ^2,
                        data,
                        adaptive=false, dt=DT,
                        dense=false)
    end
    @test_broken PNDE.fenrir_data_loglik(
        prob,
        EK0(smooth=true);
        observation_noise_cov=σ^2,
        data,
        adaptive=false, dt=DT,
        dense=false,
    )
end

@testset "Partial observations" begin
    H = [1 0;]
    data_part = (t=times, u=[H * d for d in obss])

    dalton_ll = @test_nowarn PNDE.dalton_data_loglik(
        prob,
        EK1(smooth=false);
        observation_matrix=H,
        observation_noise_cov=σ^2,
        data=data_part,
        adaptive=false, dt=DT,
        dense=false,
    )

    filtering_ll = @test_nowarn PNDE.filtering_data_loglik(
        prob,
        EK1(smooth=false);
        observation_matrix=H,
        observation_noise_cov=σ^2,
        data=data_part,
        adaptive=false, dt=DT,
        dense=false,
    )

    fenrir_ll = @test_nowarn PNDE.fenrir_data_loglik(
        prob,
        EK1(smooth=true);
        observation_matrix=H,
        observation_noise_cov=σ^2,
        data=data_part,
        adaptive=false, dt=DT,
        dense=false,
    )

    @test dalton_ll ≈ filtering_ll rtol=1e-7
    @test dalton_ll ≈ fenrir_ll rtol=1e-7
end
