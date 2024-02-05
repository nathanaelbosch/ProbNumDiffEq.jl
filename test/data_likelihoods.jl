using ProbNumDiffEq, Plots, Statistics, LinearAlgebra
import ProbNumDiffEq as PNDE
import ODEProblemLibrary: prob_ode_lotkavolterra
using Test

prob = prob_ode_lotkavolterra

# True solution
sol = solve(prob, EK1(diffusionmodel=FixedDiffusion()), abstol=1e-9, reltol=1e-9);

# Noisy observations
σ = 5e-1
times = range(prob.tspan..., length=3)[2:end];
data = [mean(sol(t)) + σ * randn(length(prob.u0)) for t in times];

DT = 2e-2

dll = @test_nowarn PNDE.dalton_data_loglik(
    prob,
    EK1(smooth=false);
    # observation_matrix=I,
    observation_noise_cov=σ^2,
    data=(t=times, u=data),
    adaptive=false, dt=DT,
    dense=false,
)

fll = @test_nowarn PNDE.filtering_data_loglik(
    prob,
    EK1(smooth=false);
    # observation_matrix=I,
    observation_noise_cov=σ^2,
    data=(t=times, u=data),
    adaptive=false, dt=DT,
    dense=false,
)

@test_broken PNDE.dalton_data_loglik(
    prob,
    EK0(smooth=false);
    # observation_matrix=I,
    observation_noise_cov=σ^2,
    data=(t=times, u=data),
    adaptive=false, dt=DT,
    dense=false,
)
@test_broken PNDE.filtering_data_loglik(
    prob,
    EK0(smooth=false);
    # observation_matrix=I,
    observation_noise_cov=σ^2,
    data=(t=times, u=data),
    adaptive=false, dt=DT,
    dense=false,
)

H = [1 0;]
@test_nowarn PNDE.dalton_data_loglik(
    prob,
    EK1(smooth=false);
    observation_matrix=H,
    observation_noise_cov=σ^2,
    data=(t=times, u=[H * d for d in data]),
    adaptive=false, dt=DT,
    dense=false,
)
@test_nowarn PNDE.filtering_data_loglik(
    prob,
    EK1(smooth=false);
    observation_matrix=H,
    observation_noise_cov=σ^2,
    data=(t=times, u=[H * d for d in data]),
    adaptive=false, dt=DT,
    dense=false,
)
