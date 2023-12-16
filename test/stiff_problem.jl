using ProbNumDiffEq
using Test
using OrdinaryDiffEq
using ODEProblemLibrary: prob_ode_vanderpol_stiff

prob = prob_ode_vanderpol_stiff
appxsol = solve(prob, RadauIIA5())
sol = solve(prob, EK1(order=3))
@test appxsol[end] ≈ sol[end] rtol = 5e-3
@test appxsol(0.5) ≈ sol(0.5).μ rtol = 5e-3
