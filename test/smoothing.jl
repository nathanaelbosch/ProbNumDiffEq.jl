using ProbNumDiffEq
using Test
using LinearAlgebra
using Statistics
using OrdinaryDiffEq
using OrdinaryDiffEqFIRK: RadauIIA5
import ODEProblemLibrary: prob_ode_lotkavolterra
using Plots

prob = prob_ode_lotkavolterra

@testset "Smoothing for small dt and large q" begin
    dt = 1e-5
    q = 8
    @test_nowarn solve(
        remake(prob, tspan=(0.0, 10dt)),
        EK0(order=q, smooth=true, diffusionmodel=FixedDiffusion()),
        adaptive=false,
        dt=dt,
    )
    @test_nowarn solve(
        remake(prob, tspan=(0.0, 10dt)),
        EK1(order=q, smooth=true, diffusionmodel=FixedDiffusion()),
        adaptive=false,
        dt=dt,
    )
end

@testset "Smooth vs. non-smooth" begin
    q = 3
    dt = 1e-2

    sol_nonsmooth =
        solve(prob, EK0(order=q, smooth=false), dense=false, adaptive=false, dt=dt)
    sol_smooth = solve(prob, EK0(order=q, smooth=true), adaptive=false, dt=dt)

    @test sol_nonsmooth.t ≈ sol_smooth.t
    @test sol_nonsmooth.u[end] == sol_smooth.u[end]
    @test sol_nonsmooth.u[end-1] != sol_smooth.u[end-1]

    plot(sol_smooth, label="smooth")
    plot!(sol_nonsmooth, label="nonsmooth")

    sol_best = solve(prob, Tsit5(), abstol=1e-12, reltol=1e-12)
    sol_best_u = sol_best.(sol_smooth.t)

    nonsmooth_errors = norm.(sol_nonsmooth.u - sol_best_u, 2)
    smooth_errors = norm.(sol_smooth.u - sol_best_u, 2)

    @test 2 * maximum(nonsmooth_errors) > maximum(smooth_errors)
    @test 2 * sum(nonsmooth_errors) > sum(smooth_errors)

    # Previously we compared the smooth and non-smooth dense output, but this
    # does not work anymore since dense output requires smoothing!
end

@testset "Backward smoothing stability at high orders (#393)" begin
    function vanderpol!(du, u, p, t)
        du[1] = u[2]
        du[2] = p[1] * ((1 - u[1]^2) * u[2] - u[1])
    end
    vdp_prob = ODEProblem(vanderpol!, [2.0, 0.0], (0.0, 2.0), [1e5])
    ref = solve(vdp_prob, RadauIIA5(), abstol=1e-14, reltol=1e-14)

    for order in [6, 7], smoother in [:mbf, :rts]
        sol = solve(
            vdp_prob, EK1(order=order, smooth=true, smoother=smoother),
            abstol=1e-10, reltol=1e-7)
        # Smoothed means/covariances should always be finite
        @test all(u -> all(isfinite, u), sol.u)
        @test all(x -> all(isfinite, x.Σ.R), sol.x_smooth)
        if smoother == :mbf
            # The √MBF regression: the dense (interpolated) smoothed mean used to blow up
            # to ~1e75 or NaN at these orders; MBF now tracks the reference solution about
            # as closely as the filtered/final estimate does. The :rts smoother instead
            # keeps the #393 behavior of main: its smoothed covariances blow up beyond
            # the filtered covariance scale at these orders, so the dense output is
            # finite but inaccurate - the accuracy checks only apply to :mbf.
            ts = range(vdp_prob.tspan..., length=1000)
            dense_err = maximum(norm(mean(sol(t)) - ref(t)) for t in ts)
            final_err = norm(sol[end] - ref[end])
            @test dense_err < 1e-3
            @test dense_err < 1e6 * final_err
        end
    end
end

@testset "Smoothed states match the backward-kernel (RTS) recursion" begin
    # The √MBF smoother must compute the exact same posterior as the RTS recursion
    # stored in `sol.backward_kernels` (both are exact for the same model), up to
    # floating-point roundoff. This guards the covariance path: a previous version
    # of the √MBF Λ update used `S_U \ H` (i.e. S_U⁻¹H) instead of `S_U' \ H`
    # (S_U⁻ᵀH) in its QR stack, which silently corrupted the smoothed covariances
    # by O(10%) while leaving the (λ-based) means essentially unaffected.
    function rts_reference(sol)
        x_smooth = [
            ProbNumDiffEq.Gaussian(copy(x.μ), ProbNumDiffEq.PSDMatrix(copy(x.Σ.R)))
            for x in sol.x_filt
        ]
        C_DxD = zero(sol.cache.C_DxD)
        C_3DxD = zero(sol.cache.C_3DxD)
        for i in (length(x_smooth)-1):-1:1
            ProbNumDiffEq.marginalize!(x_smooth[i], x_smooth[i+1],
                sol.backward_kernels[i]; C_DxD, C_3DxD)
        end
        return x_smooth
    end

    # Run both smoother implementations and check that they compute the same, exact
    # posterior as the kernel-based RTS recursion.
    for smoother in (:mbf, :rts), Alg in (EK1, EK0, DiagonalEK1)
        alg = Alg(order=3, smooth=true, smoother=smoother, save_backward_kernels=true)
        sol = solve(prob, alg, abstol=2e-2, reltol=2e-2)
        ref = rts_reference(sol)
        for i in eachindex(sol.t)
            @test sol.x_smooth[i].μ ≈ ref[i].μ rtol = 1e-8 atol = 1e-10
            @test Matrix(sol.x_smooth[i].Σ) ≈ Matrix(ref[i].Σ) rtol = 1e-8
        end
    end

    # The IOUP prior with `update_rate_parameter=true` changes its rate parameter at every
    # step; the backward smoother must use the per-step values (snapshotted and restored
    # with the smoother states), not the final one (#393-adjacent former bug).
    for smoother in (:mbf, :rts)
        alg = EK1(
            order=3, smooth=true, smoother=smoother, save_backward_kernels=true,
            prior=IOUP(3; update_rate_parameter=true))
        sol = solve(prob, alg, abstol=2e-2, reltol=2e-2)
        ref = rts_reference(sol)
        for i in eachindex(sol.t)
            @test sol.x_smooth[i].μ ≈ ref[i].μ rtol = 1e-8 atol = 1e-10
            @test Matrix(sol.x_smooth[i].Σ) ≈ Matrix(ref[i].Σ) rtol = 1e-8
        end
    end
end

@testset "MBF smoother with calibrated static diffusion" begin
    for Alg in (EK1, EK0, DiagonalEK1)
        alg_mbf = Alg(
            order=3, smooth=true, smoother=:mbf, save_backward_kernels=true,
            diffusionmodel=FixedDiffusion())
        alg_rts = Alg(
            order=3, smooth=true, smoother=:rts, save_backward_kernels=true,
            diffusionmodel=FixedDiffusion())
        sol_mbf = solve(prob, alg_mbf, abstol=2e-2, reltol=2e-2)
        sol_rts = solve(prob, alg_rts, abstol=2e-2, reltol=2e-2)
        @test length(sol_mbf.t) == length(sol_rts.t)
        for i in eachindex(sol_mbf.t)
            @test sol_mbf.x_smooth[i].μ ≈ sol_rts.x_smooth[i].μ rtol = 1e-8 atol = 1e-10
            @test Matrix(sol_mbf.x_smooth[i].Σ) ≈
                  Matrix(sol_rts.x_smooth[i].Σ) rtol = 1e-8
        end
    end
end

@testset "MBF smoother with calibrated per-dimension static diffusion" begin
    # FixedMVDiffusion only supports EK0; with it, EK0 switches from
    # IsometricKroneckerCovariance to BlockDiagonalCovariance, so this exercises the
    # BlocksOfDiagonals MBF path together with the per-dimension S_U rescaling.
    alg_mbf = EK0(
        order=3, smooth=true, smoother=:mbf,
        diffusionmodel=FixedMVDiffusion(initial_diffusion=[0.5, 2.0]))
    alg_rts = EK0(
        order=3, smooth=true, smoother=:rts, save_backward_kernels=true,
        diffusionmodel=FixedMVDiffusion(initial_diffusion=[0.5, 2.0]))
    sol_mbf = solve(prob, alg_mbf, abstol=2e-2, reltol=2e-2)
    sol_rts = solve(prob, alg_rts, abstol=2e-2, reltol=2e-2)
    @test length(sol_mbf.t) == length(sol_rts.t)
    for i in eachindex(sol_mbf.t)
        @test sol_mbf.x_smooth[i].μ ≈ sol_rts.x_smooth[i].μ rtol = 1e-8 atol = 1e-10
        @test Matrix(sol_mbf.x_smooth[i].Σ) ≈ Matrix(sol_rts.x_smooth[i].Σ) rtol = 1e-8
    end
end

@testset "degenerate (zero-covariance) steps with observation noise" begin
    prob = prob_ode_lotkavolterra
    # check test/observation_noise.jl for the accepted pn_observation_noise format
    # (Diagonal / Matrix / PSDMatrix are supported)
    alg = EK1(
        initialization=TaylorModeInit(3),
        diffusionmodel=FixedDiffusion(0, false),          # zero initial diffusion, no calibration
        pn_observation_noise=Diagonal(fill(0.1, 2)),
        smooth=true,
    )
    sol = solve(prob, alg)
    # The filter skipped every update (Σ_pred ≡ 0), so there is no measurement
    # information to smooth with: every smoother state must be `nothing`.
    @test all(ss -> ss === nothing, sol.smoother_states)   # currently fails: stores SmootherStates
    @test all(x -> all(isfinite, x.μ), sol.x_smooth)
    @test all(x -> all(isfinite, x.Σ.R), sol.x_smooth)
end

@testset "smoother_states stay index-aligned with the saved times" begin
    # (a) save_end=false: final savevalues! runs without an underlying save
    sol = solve(prob, EK1(); save_end=false)
    @test length(sol.smoother_states) == length(sol.t) - 1

    # (b) two discrete callbacks firing at every step (duplicate-time saves;
    #     each step produces one no-save custom run followed by force-saves)
    cb = CallbackSet(
        DiscreteCallback((u, t, integ) -> true, integ -> nothing),
        DiscreteCallback((u, t, integ) -> true, integ -> nothing),
    )
    sol = solve(prob, EK1(); callback=cb)
    @test length(sol.smoother_states) == length(sol.t) - 1
end

@testset "backward kernels are only computed when consumed" begin
    prob = prob_ode_lotkavolterra

    # :rts without smooth => nothing consumes the kernels: don't compute or store them
    sol = solve(prob, EK1(smoother=:rts, smooth=false); dense=false)
    @test length(sol.backward_kernels) == 0

    # independent opt-in still works without smoothing
    sol = solve(prob, EK1(save_backward_kernels=true, smooth=false); dense=false)
    @test length(sol.backward_kernels) == length(sol.t) - 1

    # and with smoothing (unchanged behavior)
    sol = solve(prob, EK1(smoother=:rts, smooth=true); dense=false)
    @test length(sol.backward_kernels) == length(sol.t) - 1
    sol = solve(prob, EK1(smooth=true); dense=false)             # default :mbf
    @test length(sol.backward_kernels) == 0
end
