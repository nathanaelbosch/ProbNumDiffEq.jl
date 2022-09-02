using ProbNumDiffEq
using Test
import ODEProblemLibrary: prob_ode_fitzhughnagumo

prob = prob_ode_fitzhughnagumo
prob = remake(prob, u0=big.(prob.u0))
sol = solve(prob, EK0(order=3))
@test eltype(eltype(sol.u)) == BigFloat
@test eltype(eltype(sol.pu.μ)) == BigFloat
@test eltype(eltype(sol.pu.Σ)) == BigFloat
@test sol isa ProbNumDiffEq.ProbODESolution
