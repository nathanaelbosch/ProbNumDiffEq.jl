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

@testset "ManifoldUpdateCallback" begin
    sol1 = solve(prob, EK1(order=3))

    E(u) = [dot(u, u) - 2]
    @test_broken solve(prob, EK0(order=3), callback=ManifoldUpdateCallback(E))
    @test_nowarn solve(prob, EK1(order=3), callback=ManifoldUpdateCallback(E))
    sol2 = solve(prob, EK1(order=3), callback=ManifoldUpdateCallback(E))

    @test E(sol1.u[end]) .^ 2 > E(sol2.u[end]) .^ 2

    err1 = sol1.u[end] .- appxsol.u[end]
    err2 = sol2.u[end] .- appxsol.u[end]
    @test all(err1 .^ 2 > err2 .^ 2)
end
