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
observe(H) = (t=times, u=[H * x for x in obss])
H_part = [1 0;]
data_part = observe(H_part)

DT = 2e-2

# A linear ODE with a diagonal Jacobian, on which the `EK1` and `DiagonalEK1` coincide
prob_lin = ODEProblem((du, u, p, t) -> (du .= -u), [1.0, 2.0, 3.0], (0.0, 5.0), 1.0)

function compare_data_likelihoods(alg; problem=prob, kwargs...)
    dalton_ll = @test_nowarn PNDE.dalton_data_loglik(
        problem, remake(alg, smooth=false); kwargs...)
    filtering_ll = @test_nowarn PNDE.filtering_data_loglik(
        problem, remake(alg, smooth=false); kwargs...)
    fenrir_ll = PNDE.fenrir_data_loglik(
        problem, remake(alg, smooth=true); kwargs...,
    )
    @test dalton_ll ≈ filtering_ll rtol = 1e-6
    @test dalton_ll ≈ fenrir_ll rtol = 1e-6
    return dalton_ll
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
        EK0(diffusionmodel=FixedMVDiffusion(rand(2), false)),
        EK0(diffusionmodel=DynamicMVDiffusion()),
        EK0(prior=IOUP(3, -1)),
        EK0(prior=Matern(3, 1.5)),
        # EK1
        EK1(),
        EK1(diffusionmodel=FixedDiffusion()),
        EK1(diffusionmodel=FixedMVDiffusion(rand(2), false)),
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
    # EK0 with a multivariate diffusion uses a block-diagonal covariance structure
    # as well, so partial observations work the same way as with the DiagonalEK1
    for alg in (EK1(), DiagonalEK1(), EK0(diffusionmodel=FixedMVDiffusion(rand(2), false)))
        compare_data_likelihoods(alg; kwargs..., observation_matrix=H_part, data=data_part)
    end
    # Non-unit row scalings need no special handling: the scale is part of the
    # observation matrix, so the noise covariance scales with it
    H2 = [2 0]
    data_scaled = observe(H2)
    for alg in (DiagonalEK1(), EK1())
        compare_data_likelihoods(
            alg; kwargs...,
            observation_matrix=H2, observation_noise_cov=4σ^2, data=data_scaled)
    end
    # Observation noise covariance types, like in the testset below, but for partial
    # observations; the per-observation noise variances are extracted from the matrix
    for R in (σ^2 * I, Diagonal([σ^2]), fill(σ^2, 1, 1), PSDMatrix(fill(σ, 1, 1)))
        compare_data_likelihoods(
            DiagonalEK1(); kwargs...,
            observation_matrix=H_part, observation_noise_cov=R, data=data_part)
    end
    # With a block-diagonal covariance, observation matrices that select all dimensions,
    # e.g. permutations, are handled block-wise as well
    P = [0 1; 1 0]
    data_perm = observe(P)
    compare_data_likelihoods(
        DiagonalEK1(); kwargs..., observation_matrix=P, data=data_perm)
    @test PNDE.dalton_data_loglik(
        prob, remake(DiagonalEK1(), smooth=false); kwargs...,
        observation_matrix=P, data=data_perm,
    ) ≈ PNDE.dalton_data_loglik(prob, remake(DiagonalEK1(), smooth=false); kwargs...)
    # On `prob_lin`, the block-wise update of the `DiagonalEK1` matches the dense one of
    # the `EK1`, also for scaled observations in unsorted order
    H_lin = [0 0 2; 1 0 0]
    data_lin = (t=[1.0, 4.0], u=[H_lin * randn(3) for _ in 1:2])
    ll_ek1, ll_diag = (
        compare_data_likelihoods(
            alg; kwargs..., problem=prob_lin, observation_matrix=H_lin, data=data_lin)
        for alg in (EK1(), DiagonalEK1())
    )
    @test ll_diag ≈ ll_ek1 rtol = 1e-6
end

@testset "Scalar observation matrices" begin
    @testset "$alg" for alg in (EK0(), EK1(), DiagonalEK1())
        lls = [
            compare_data_likelihoods(
                alg;
                kwargs...,
                observation_matrix=M,
                data=observe(2I),
            )
            for M in (2.0, 2I)
        ]
        @test lls[1] ≈ lls[2]
    end
end

function test_data_likelihoods_throw(E, alg; problem=prob, kwargs...)
    @test_throws E PNDE.dalton_data_loglik(problem, remake(alg, smooth=false); kwargs...)
    @test_throws E PNDE.filtering_data_loglik(problem, remake(alg, smooth=false); kwargs...)
    @test_throws E PNDE.fenrir_data_loglik(problem, remake(alg, smooth=true); kwargs...)
end

@testset "Bad observation matrices" begin
    # Partial observations with `DiagonalEK1` require dimension-selection observation
    # matrices: each row must have exactly one nonzero entry (any scaling is fine);
    # rows that mix dimensions can not be handled block-wise
    H = [1 1]
    test_data_likelihoods_throw(
        ArgumentError, DiagonalEK1(); kwargs..., observation_matrix=H, data=observe(H))

    # Observing the same dimension twice requires at least three state dimensions
    times3 = [1.0, 4.0]
    H3 = [1 0 0; 1 0 0]
    data3 = (t=times3, u=[H3 * randn(3) for _ in times3])
    test_data_likelihoods_throw(
        ArgumentError, DiagonalEK1(); kwargs...,
        problem=prob_lin, observation_matrix=H3, data=data3)

    # EK0 with the default isometric-kronecker covariance structure only supports `I` as
    # observation matrix, and isotropic noise
    P = [0 1; 1 0]
    for bad in (
        (observation_matrix=H_part, data=data_part),
        (observation_matrix=P, data=observe(P)),
        (observation_noise_cov=Diagonal([σ^2, 2σ^2]),),
    )
        test_data_likelihoods_throw(ArgumentError, EK0(); kwargs..., bad...)
    end

    # Observation noise covariances which can not be decomposed into one noise
    # variance per observation (e.g. correlated noise) are not supported
    H2 = [1 0 0; 0 1 0]
    data2 = (t=times3, u=[H2 * randn(3) for _ in times3])
    test_data_likelihoods_throw(
        ArgumentError, DiagonalEK1(); kwargs...,
        problem=prob_lin, observation_matrix=H2, observation_noise_cov=[1.0 0.5; 0.5 1.0],
        data=data2)

    # With every solver, the observation matrix needs one row per data entry (the default
    # `I` has `d`), one column per ODE dimension, at most `d` rows and no zero rows, and
    # the noise covariance one row and column per data entry
    H_tall = [1 0; 0 1; 1 0]
    @testset "$alg" for alg in (EK0(), EK1(), DiagonalEK1())
        for (E, bad) in (
            (DimensionMismatch, (observation_matrix=[1 0], data=data)),
            (DimensionMismatch, (observation_matrix=I, data=data_part)),
            (
                DimensionMismatch,
                (observation_matrix=H_part, data=data_part,
                    observation_noise_cov=σ^2 * I(2)),
            ),
            (
                ArgumentError,
                (observation_matrix=[1 0 0], data=(t=times, u=[[0.0] for _ in times])),
            ),
            (ArgumentError, (observation_matrix=H_tall, data=observe(H_tall))),
            (ArgumentError, (observation_matrix=[0 0], data=observe([0 0]))),
            (ArgumentError, (observation_matrix=Diagonal([1, 0]), data=data)),
        )
            test_data_likelihoods_throw(E, alg; kwargs..., bad...)
        end
    end
end

@testset "Second-order ODEs" begin
    # The observation matrix acts on `u` only, not on `du`
    prob2 = SecondOrderODEProblem(
        (ddu, du, u, p, t) -> (ddu .= -p .* u), [0.0, 1.0], [1.0, 0.0], (0.0, 10.0), 1.0,
    )
    obss2 = [randn(2) for _ in times]
    @testset "H = $H" for (H, algs) in (
        (I, (EK0(), EK1(), DiagonalEK1())),
        (
            [0 1],
            (EK1(), DiagonalEK1(), EK0(diffusionmodel=FixedMVDiffusion(ones(2), false))),
        ),
    )
        data2 = (t=times, u=[H * x for x in obss2])
        @testset "$alg" for alg in algs
            compare_data_likelihoods(
                alg; kwargs..., problem=prob2, observation_matrix=H, data=data2)
        end
    end
    data_du = (t=times, u=[randn(1) for _ in times])
    test_data_likelihoods_throw(
        ArgumentError, EK1(); kwargs..., problem=prob2, observation_matrix=[0 0 1 0],
        data=data_du)
end

@testset "Observation noise types: $(typeof(Σ))" for Σ in (
    σ^2,
    σ^2 * I,
    σ^2 * I(2),
    σ^2 * Eye(2),
    Diagonal([σ^2 0; 0 2σ^2]),
    [σ^2 0; 0 2σ^2],
    (A=randn(2, 2); A'A),
    (PSDMatrix(randn(2, 2))),
)
    @testset "$alg" for alg in (EK0(), DiagonalEK1(), EK1())
        if alg isa EK0 && Σ isa AbstractMatrix &&
           !(isdiag(Matrix(Σ)) && allequal(diag(Matrix(Σ))))
            continue
        end
        if alg isa DiagonalEK1 && Σ isa AbstractMatrix && !isdiag(Matrix(Σ))
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

# Full observations with all solvers; partial ones with those that support them
H_algs = (
    (I, (EK0(), EK1(), DiagonalEK1())),
    ([1 0 0; 0 1 0], (EK1(), DiagonalEK1())),
)

@testset "Data that ends before the time span" begin
    obss_lin = [randn(3) for _ in 1:2]
    @testset "H = $H" for (H, algs) in H_algs
        data_lin = (t=[1.0, 4.0], u=[H * x for x in obss_lin])
        @testset "$alg" for alg in algs
            compare_data_likelihoods(
                alg; kwargs..., problem=prob_lin, observation_matrix=H, data=data_lin)
        end
    end
end

@testset "Data at the initial time" begin
    # The initial value is known exactly, so a data point at `t0` leaves the state
    # unchanged and adds its log-likelihood under `N(H u0, R)`
    y0 = [1.0, 0.0, 3.0]
    obss_later = [randn(3) for _ in 1:2]
    @testset "H = $H" for (H, algs) in H_algs
        r = H * (y0 - prob_lin.u0)
        ll_t0 = -(sum(abs2, r) / σ^2 + length(r) * log(2π * σ^2)) / 2
        data_later = (t=[1.0, 4.0], u=[H * x for x in obss_later])
        data_with_t0 = (t=[0.0; data_later.t], u=[[H * y0]; data_later.u])
        @testset "$alg" for alg in algs
            compare_data_likelihoods(
                alg; kwargs..., problem=prob_lin, observation_matrix=H, data=data_with_t0)
            for loglik in (
                PNDE.dalton_data_loglik, PNDE.filtering_data_loglik,
                PNDE.fenrir_data_loglik,
            )
                smooth = loglik === PNDE.fenrir_data_loglik
                ll =
                    data -> loglik(
                        prob_lin, remake(alg; smooth); kwargs..., observation_matrix=H,
                        data)
                @test ll((t=[0.0], u=[H * y0])) ≈ ll_t0
                @test ll(data_with_t0) ≈ ll_t0 + ll(data_later)
            end
        end
    end
end

@testset "Non-positive observation noise" begin
    for R in (0.0, 0.0I, Diagonal([σ^2, 0.0]), [σ^2 0; 0 0], PSDMatrix(zeros(2, 2)))
        test_data_likelihoods_throw(
            ArgumentError,
            EK1();
            kwargs...,
            observation_noise_cov=R,
        )
    end
end
