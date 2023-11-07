using ProbNumDiffEq
using OrdinaryDiffEq
using Test

du0 = [0.0]
u0 = [2.0]
tspan = (0.0, 0.1)
p = [1e1]

function vanderpol!(ddu, du, u, p, t)
    μ = p[1]
    @. ddu = μ * ((1 - u^2) * du - u)
end
const prob_iip = SecondOrderODEProblem(vanderpol!, du0, u0, tspan, p)

function vanderpol(du, u, p, t)
    μ = p[1]
    ddu = μ .* ((1 .- u .^ 2) .* du .- u)
    return ddu
end
const prob_oop = SecondOrderODEProblem(vanderpol, du0, u0, tspan, p)

const appxsol = solve(prob_iip, Tsit5(), abstol=1e-7, reltol=1e-7)

ALGS = (
    EK0(),
    EK1(),
)

@testset "IIP" begin
    for alg in ALGS
        @testset "$alg" begin
            sol = solve(prob_iip, alg)
            # @test sol isa ProbNumDiffEq.ProbODESolution
            # @test sol.u[end] ≈ appxsol.u[end] rtol = 1e-3
        end
    end
end

@testset "OOP" begin
    for alg in ALGS
        @testset "$alg" begin
            sol = solve(prob_oop, alg)
            @test sol isa ProbNumDiffEq.ProbODESolution
            @test sol.u[end] ≈ appxsol.u[end] rtol = 1e-3
        end
    end
end

_name(structinstance) = typeof(structinstance).name.wrapper
@testset "Diffusion: $(_name(DIFFUSION))" for DIFFUSION in (
    FixedDiffusion(), DynamicDiffusion(), FixedMVDiffusion(), DynamicMVDiffusion())
    @test_nowarn solve(
        prob_iip,
        EK0(diffusionmodel=FixedDiffusion(), smooth=false),
        save_everystep=false,
        dense=false,
    )
end

@testset "Initialization: $(_name(INIT))" for INIT in (
    TaylorModeInit(3), ForwardDiffInit(3), SimpleInit(), ClassicSolverInit())
    @test_nowarn solve(
        prob_iip,
        EK1(initialization=INIT, smooth=false),
        save_everystep=false,
        dense=false,
    )
end
