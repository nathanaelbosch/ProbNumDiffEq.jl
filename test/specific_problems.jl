# Everytime I encounter something that raises some error and I fix it, I should add that
# specific problem to this list to make sure, that this specific run then works without
# bugs.
using ProbNumDiffEq
using ModelingToolkit
using Test
using LinearAlgebra
using UnPack
using ParameterizedFunctions
using OrdinaryDiffEq
using Plots

using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary:
    prob_ode_fitzhughnagumo, prob_ode_vanderpol_stiff, prob_ode_2Dlinear, prob_ode_linear

@testset "Problem with analytic solution" begin
    linear(u, p, t) = p .* u
    linear_analytic(u0, p, t) = @. u0 * exp(p * t)
    prob =
        ODEProblem(ODEFunction(linear, analytic=linear_analytic), [1 / 2], (0.0, 1.0), 1.01)

    @test solve(prob, EK0()) isa ProbNumDiffEq.ProbODESolution
    sol = solve(prob, EK0())
    @test sol.errors isa Dict
    @test all(haskey.(Ref(sol.errors), (:l∞, :l2, :final)))
end

@testset "Matrix-Valued Problem" begin
    prob = remake(prob_ode_2Dlinear, u0=rand(2, 2))

    @testset "$alg" for alg in [EK0(), EK1()]
        sol = solve(prob, alg)
        @test sol isa ProbNumDiffEq.ProbODESolution

        @test length(sol.u[1]) == length(sol.pu.μ[1])
        @test sol.u[1][:] == sol.pu.μ[1]
        @test sol.u ≈ sol.u_analytic rtol = 1e-4
        @test plot(sol) isa AbstractPlot
    end
end

@testset "scalar-valued problem" begin
    @testset "$alg" for alg in [EK0(), EK1()]
        @test_broken solve(prob_ode_linear, alg) isa ProbNumDiffEq.ProbODESolution
    end
end

@testset "Stiff Vanderpol" begin
    prob = prob_ode_vanderpol_stiff
    # prob = ODEProblem(modelingtoolkitize(prob), prob.u0, prob.tspan, jac=true)
    @test solve(prob, EK1(order=3)) isa ProbNumDiffEq.ProbODESolution
end

@testset "Big Float" begin
    prob = prob_ode_fitzhughnagumo
    prob = remake(prob, u0=big.(prob.u0))
    sol = solve(prob, EK0(order=3))
    @test eltype(eltype(sol.u)) == BigFloat
    @test eltype(eltype(sol.pu.μ)) == BigFloat
    @test eltype(eltype(sol.pu.Σ)) == BigFloat
    @test sol isa ProbNumDiffEq.ProbODESolution
end

@testset "OOP problem" begin
    f(u, p, t) = p .* u .* (1 .- u)
    prob = ODEProblem(f, [1e-1], (0.0, 2.0), [3.0])
    @testset "without jacobian" begin
        # first without defined jac
        @test solve(prob, EK0(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob, EK1(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob, EK1(order=4, initialization=ClassicSolverInit())) isa
              ProbNumDiffEq.ProbODESolution
    end
    @testset "with jacobian" begin
        # now with defined jac
        prob = ODEProblem(modelingtoolkitize(prob), prob.u0, prob.tspan, jac=true)
        @test solve(prob, EK0(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob, EK1(order=4)) isa ProbNumDiffEq.ProbODESolution
        @test solve(prob, EK1(order=4, initialization=ClassicSolverInit())) isa
              ProbNumDiffEq.ProbODESolution
    end
end

@testset "Callback: Harmonic Oscillator with condition on E=2" begin
    u0 = ones(2)
    function harmonic_oscillator(du, u, p, t)
        du[1] = u[2]
        return du[2] = -u[1]
    end
    prob = ODEProblem(harmonic_oscillator, u0, (0.0, 100.0))

    function Callback()
        function affect!(integ)
            @unpack x_filt, Proj, PI, E0 = integ.cache

            x = x_filt

            m, P = x.μ, x.Σ

            m0, P0 = E0 * m, ProbNumDiffEq.X_A_Xt(P, E0)

            e = m0'm0
            H = 2m0'E0
            S = H * P * H'

            S_inv = inv(S)
            K = P * H' * S_inv

            mnew = m + K * (2 .- e)
            Pnew = ProbNumDiffEq.X_A_Xt(P, (I - K * H)) # + X_A_Xt(R, K)

            # @info m P e S K mnew
            copy!(m, mnew)
            return copy!(P, Pnew)
        end
        condtion = (t, u, integrator) -> true
        save_positions = (true, true)
        return DiscreteCallback(condtion, affect!, save_positions=save_positions)
    end

    @test solve(prob, EK0(order=3)) isa ProbNumDiffEq.ProbODESolution
    @test solve(prob, EK0(order=3), callback=Callback()) isa ProbNumDiffEq.ProbODESolution
end

@testset "ManifoldUpdate callback test" begin
    # Again: Harmonic Oscillator with condition on E=2
    u0 = ones(2)
    function harmonic_oscillator(du, u, p, t)
        du[1] = u[2]
        return du[2] = -u[1]
    end
    prob = ODEProblem(harmonic_oscillator, u0, (0.0, 10.0))
    appxsol = solve(prob, Vern9(), abstol=1e-10, reltol=1e-10)

    E(u) = [dot(u, u) - 2]

    sol1 = solve(prob, EK0(order=3))
    @test sol1 isa ProbNumDiffEq.ProbODESolution
    sol2 = solve(prob, EK0(order=3), callback=ManifoldUpdate(E))
    @test sol2 isa ProbNumDiffEq.ProbODESolution

    @test E(sol1[end]) .^ 2 > E(sol2[end]) .^ 2

    err1 = sol1[end] .- appxsol[end]
    err2 = sol2[end] .- appxsol[end]
    @test all(err1 .^ 2 > err2 .^ 2)
end

@testset "Problem definition with ParameterizedFunctions.jl" begin
    f = @ode_def LotkaVolterra begin
        dx = a * x - b * x * y
        dy = -c * y + d * x * y
    end a b c d
    p = [1.5, 1, 3, 1]
    tspan = (0.0, 1.0)
    u0 = [1.0, 1.0]
    prob = ODEProblem(f, u0, tspan, p)
    @test solve(prob, EK1(order=3)) isa ProbNumDiffEq.ProbODESolution
end

@testset "Robertson in mass-matrix-ODE form" begin
    function rober(du, u, p, t)
        y₁, y₂, y₃ = u
        k₁, k₂, k₃ = p
        du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
        du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
        du[3] = y₁ + y₂ + y₃ - 1
        return nothing
    end
    M = [
        1.0 0 0
        0 1.0 0
        0 0 0
    ]
    f = ODEFunction(rober, mass_matrix=M)
    prob = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4))

    sol1 = solve(prob, EK1(order=3))
    sol2 = solve(prob, RadauIIA5())
    @test sol1[end] ≈ sol2[end] rtol = 1e-5
end

@testset "EK1 Jacobian computation" begin
    prob = prob_ode_fitzhughnagumo
    @assert isnothing(prob.f.jac)

    # make sure that the kwarg works
    sol1 = solve(prob, EK1())
    sol2 = solve(prob, EK1(autodiff=false))
    @test sol2 isa ProbNumDiffEq.ProbODESolution

    # check that forwarddiff leads to a smaller nf than finite diff
    @test sol1.destats.nf < sol2.destats.nf

    # use the EK1 on a non-forwarddiffable function
    # TODO
end
