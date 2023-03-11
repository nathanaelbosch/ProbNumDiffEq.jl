#=
In OrdinaryDiffEq.jl, implicit solvers have some keyword arguments defined
that control how the Jacobian is computed. This is what we test here.
=#
using ProbNumDiffEq
using Test
using OrdinaryDiffEq
import ODEProblemLibrary: prob_ode_fitzhughnagumo

prob = prob_ode_fitzhughnagumo
@assert isnothing(prob.f.jac)

# make sure that the kwarg works
sol1 = solve(prob, EK1())
sol2 = solve(prob, EK1(autodiff=false))
@test sol2 isa ProbNumDiffEq.ProbODESolution

# check that forwarddiff leads to a smaller nf than finite diff
@test sol1.stats.nf < sol2.stats.nf
