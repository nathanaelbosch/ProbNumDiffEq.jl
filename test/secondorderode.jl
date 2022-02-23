using ProbNumDiffEq
using Test

du0 = [0.0]
u0 = [2.0]
tspan = (0.0, 6.3)
p = [1e1]

function vanderpol!(ddu, du, u, p, t)
    μ = p[1]
    @. ddu = μ * ((1 - u^2) * du - u)
end
prob_iip = SecondOrderODEProblem(vanderpol!, du0, u0, tspan, p)

# function vanderpol(du, u, p, t)
#     μ = p[1]
#     ddu = μ .* ((1 .- u .^ 2) .* du .- u)
#     return ddu
# end
# prob_oop = SecondOrderODEProblem(vanderpol, du0, u0, tspan, p)

appxsol = solve(prob_iip, Tsit5(), abstol=1e-7, reltol=1e-7)

@testset "IIP" begin
    for Alg in (EK0, EK1)
        @testset "$Alg" begin
            @test solve(prob_iip, Alg()) isa ProbNumDiffEq.ProbODESolution
            @test solve(prob_iip, Alg()).u[end] ≈ appxsol.u[end] rtol = 1e-3
        end
    end
end

# @testset "OOP" begin
#     for Alg in (EK0, EK1)
#         @testset "$Alg" begin
#             @test solve(prob_oop, Alg()) isa ProbNumDiffEq.ProbODESolution
#             @test solve(prob_oop, Alg()).u[end] ≈ appxsol.u[end] rtol = 1e-3
#         end
#     end
# end

@testset "ClassicSolverInit for SecondOrderODEProblems" begin
    @test_broken solve(prob_iip, EK1(initialization=ClassicSolverInit())) isa
                 ProbNumDiffEq.ProbODESolution
end

@testset "Fixed Diffusion" begin
    @test solve(prob_iip, EK0(diffusionmodel=FixedDiffusion())) isa
          ProbNumDiffEq.ProbODESolution
end
