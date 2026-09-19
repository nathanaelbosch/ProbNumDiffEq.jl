using ProbNumDiffEq
using ModelingToolkit
using Test
using LinearAlgebra
using FiniteDifferences
using ForwardDiff
import SciMLStructures
# using ReverseDiff
# using Zygote

import ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_lotkavolterra

# Helper function to convert a regular ODE problem to MTK form with symbolic maps
function make_mtk_problem(_prob)
    sys = structural_simplify(modelingtoolkitize(_prob))
    unknowns_list = ModelingToolkit.unknowns(sys)
    params_list = ModelingToolkit.parameters(sys)
    u0_map = Dict(unknowns_list[i] => _prob.u0[i] for i in eachindex(_prob.u0))
    p_vals = collect(_prob.p)
    p_map = Dict(params_list[i] => p_vals[i] for i in eachindex(p_vals))
    return ODEProblem(sys, merge(u0_map, p_map), _prob.tspan; jac=true)
end

@testset "solver: $ALG" for ALG in (EK0, EK1, DiagonalEK1)
    _prob = prob_ode_fitzhughnagumo
    prob = make_mtk_problem(_prob)
    ps = ModelingToolkit.parameter_values(prob)
    ps = SciMLStructures.replace(SciMLStructures.Tunable(), ps, [1.0, 2.0, 3.0, 4.0])
    prob = remake(prob, p=ps)

    function param_to_loss(p)
        ps = ModelingToolkit.parameter_values(prob)
        ps = SciMLStructures.replace(SciMLStructures.Tunable(), ps, p)
        sol = solve(
            remake(prob, p=ps),
            ALG(order=3, smooth=false),
            sensealg=SensitivityADPassThrough(),
            abstol=1e-6,
            reltol=1e-5,
            save_everystep=false,
            dense=false,
        )
        return norm(sol.u[end])  # Dummy loss
    end
    function startval_to_loss(u0)
        sol = solve(
            remake(prob, u0=u0),
            ALG(order=3, smooth=false),
            sensealg=SensitivityADPassThrough(),
            abstol=1e-6,
            reltol=1e-5,
            save_everystep=false,
            dense=false,
        )
        return norm(sol.u[end])  # Dummy loss
    end

    p, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)

    #dldp = FiniteDiff.finite_difference_gradient(param_to_loss, p)
    #dldu0 = FiniteDiff.finite_difference_gradient(startval_to_loss, prob.u0)
    # For some reason FiniteDiff.jl is not working anymore so we use FiniteDifferences.jl:
    dldp = grad(central_fdm(5, 1), param_to_loss, p)[1]
    dldu0 = grad(central_fdm(5, 1), startval_to_loss, prob.u0)[1]

    @testset "ForwardDiff.jl" begin
        @test ForwardDiff.gradient(param_to_loss, p) ≈ dldp rtol = 1e-2
        @test ForwardDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0 rtol = 5e-2
    end

    # @testset "ReverseDiff.jl" begin
    #     @test_broken ReverseDiff.gradient(param_to_loss, prob.p) ≈ dldp
    #     @test_broken ReverseDiff.gradient(startval_to_loss, prob.u0) ≈ dldu0
    # end

    # @testset "Zygote.jl" begin
    #     @test_broken Zygote.gradient(param_to_loss, prob.p) ≈ dldp
    #     @test_broken Zygote.gradient(startval_to_loss, prob.u0) ≈ dldu0
    # end
end

@testset "Smoothed means under AD: MBF and RTS smoothers agree" begin
    # Regression test: `_extract_measurement_chol` used to read `cache.C_dxd` assuming
    # `update!` had factorized it in place, which is false for ForwardDiff.Dual eltypes
    # (see `make_hermitian_if_fowarddiff`). This corrupted the MBF smoother's saved
    # Cholesky factors and hence the smoothed means under AD. EK1 with d>=2 is the
    # affected configuration; EK0/DiagonalEK1 and d==1 EK1 use a scalar shortcut that is
    # unaffected. Only smoothed means are checked here (not covariances): smoothed-
    # covariance gradients have a separate, pre-existing issue, see the testset below.
    prob = prob_ode_lotkavolterra

    function smoothed_sum(u0, alg)
        sol = solve(
            remake(prob, u0=u0), alg, abstol=1e-6, reltol=1e-6, dense=false,
            initializealg=SimpleInit())
        return sum(norm.(sol.u[2:(end-1)]))
    end
    u0 = [1.0, 1.0]

    g_mbf = ForwardDiff.gradient(u -> smoothed_sum(u, EK1(order=3)), u0)
    g_rts = ForwardDiff.gradient(u -> smoothed_sum(u, EK1(order=3, smoother=:rts)), u0)

    @test g_mbf ≈ g_rts rtol = 1e-4
end

@testset "Smoothed covariances under AD: NaN partials with :mbf (known limitation)" begin
    # `_hyperbolic_qr!` (used by the MBF backward step to recover the smoothed
    # covariance square-root factor) produces correct *values* under ForwardDiff.Dual
    # eltypes, but NaN *partials* for losses involving the smoothed covariances. This is
    # intrinsic to the hyperbolic rotation's conditioning (~1/α for small J-norm α, see
    # `_hyperbolic_qr!`/`_mbf_recover_cov` docstrings) and is not fixable without
    # degrading the Float64 numerics. The RTS smoother is unaffected.
    prob = prob_ode_lotkavolterra

    function smoothed_cov_loss(u0, alg)
        sol = solve(
            remake(prob, u0=u0), alg, abstol=1e-6, reltol=1e-6, dense=false,
            initializealg=SimpleInit())
        return sum(
            sum(abs2, x.μ) + sum(abs2, Matrix(x.Σ)) for x in sol.x_smooth[2:(end-1)]
        )
    end
    u0 = [1.0, 1.0]

    g_mbf = ForwardDiff.gradient(u -> smoothed_cov_loss(u, EK1(order=3)), u0)
    g_rts = ForwardDiff.gradient(u -> smoothed_cov_loss(u, EK1(order=3, smoother=:rts)), u0)

    # Diagnosis (see `_hyperbolic_qr!`/`_mbf_backward_step!` docstrings in
    # src/integrator_utils.jl): the hyperbolic rotation used to recover the smoothed
    # covariance is ~1/α-conditioned in a column's J-norm α; as α → 0⁺ (short of the
    # degenerate-column cutoff), the ForwardDiff partials of α (and downstream β, τ)
    # diverge and cancel catastrophically, producing NaN partials -- confirmed
    # analytically to grow unboundedly as α → 0⁺ well before the cutoff, i.e. an
    # intrinsic conditioning property of the rotation, not an artifact of the `max(·, 0)`
    # clamp kink. No fix was found that preserves the Float64 numerics.
    @test_broken all(isfinite, g_mbf)
    @test_broken isapprox(g_mbf, g_rts; rtol=1e-3)
end
