using ProbNumDiffEq
using OrdinaryDiffEq
using LinearAlgebra
using SimpleUnPack: @unpack
using Test

u0 = ones(2)
function harmonic_oscillator(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end
prob = ODEProblem(harmonic_oscillator, u0, (0.0, 10.0))
appxsol = solve(prob, Vern9(), reltol=1e-9, abstol=1e-9)

@testset "Custom callback" begin
    function CustomCallback()
        function affect!(integ)
            @unpack x_filt, Proj, PI, E0 = integ.cache

            x = x_filt

            m, P = x.μ, x.Σ

            m0, P0 = E0 * m, ProbNumDiffEq.X_A_Xt(P, E0)

            e = m0'm0
            H = 2m0'E0
            SR = P.R * H'
            S = SR'SR

            K = P.R' * (P.R * (H' / S))

            mnew = m + K * (2 .- e)
            Pnew = ProbNumDiffEq.X_A_Xt(P, (I - K * H)) # + X_A_Xt(R, K)

            # @info m P e S K mnew
            copy!(m, mnew)
            copy!(P, Pnew)
            return nothing
        end
        condtion = (t, u, integrator) -> true
        save_positions = (true, true)
        return DiscreteCallback(condtion, affect!, save_positions=save_positions)
    end

    @test_nowarn solve(prob, EK1(order=3))
    @test_broken solve(prob, EK0(order=3), callback=CustomCallback())
    @test_nowarn solve(prob, EK1(order=3), callback=CustomCallback())
end

@testset "ManifoldUpdate callback" begin
    sol1 = solve(prob, EK1(order=3))

    E(u) = [dot(u, u) - 2]
    @test_broken solve(prob, EK0(order=3), callback=ManifoldUpdate(E))
    @test_nowarn solve(prob, EK1(order=3), callback=ManifoldUpdate(E))
    sol2 = solve(prob, EK1(order=3), callback=ManifoldUpdate(E))

    @test E(sol1.u[end]) .^ 2 > E(sol2.u[end]) .^ 2

    err1 = sol1.u[end] .- appxsol.u[end]
    err2 = sol2.u[end] .- appxsol.u[end]
    @test all(err1 .^ 2 > err2 .^ 2)
end

@testset "ManifoldUpdate with multiple constraints" begin
    function two_oscillators(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
        du[3] = u[4]
        du[4] = -u[3]
        return nothing
    end
    prob2 = ODEProblem(two_oscillators, ones(4), (0.0, 10.0))
    E2(u) = [u[1]^2 + u[2]^2 - 2; u[3]^2 + u[4]^2 - 2]

    sol1 = solve(prob2, EK1(order=3))
    sol2 = solve(prob2, EK1(order=3), callback=ManifoldUpdate(E2))
    @test all(E2(sol1.u[end]) .^ 2 .> E2(sol2.u[end]) .^ 2)
end

@testset "ManifoldUpdate residual shape errors" begin
    E_padded(u) = [dot(u, u) - 2; 0]
    @test_throws ArgumentError solve(
        prob, EK1(order=3), callback=ManifoldUpdate(E_padded))

    E_toolong(u) = [dot(u, u) - 2; u[1]; u[2]]
    @test_throws DimensionMismatch solve(
        prob, EK1(order=3), callback=ManifoldUpdate(E_toolong))
end

@testset "ManifoldUpdate allocations" begin
    E(u) = [dot(u, u) - 2]
    kwargs = (adaptive=false, dt=0.05, callback=ManifoldUpdate(E))
    solve(prob, EK1(order=3); kwargs...)  # compile
    allocs_with = @allocated solve(prob, EK1(order=3); kwargs...)
    solve(prob, EK1(order=3), adaptive=false, dt=0.05)  # compile
    allocs_without = @allocated solve(prob, EK1(order=3), adaptive=false, dt=0.05)
    # Loose bound; the callback used to allocate ~50x more than the solve itself
    @test allocs_with < 3 * allocs_without
end
