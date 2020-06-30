using ProbNumODE


# Current code
prob = fitzhugh_nagumo()
psol = prob_solve(prob, 0.05);
@show length(psol.t)


sol = solve(prob, ODEFilter(), dt=0.05)
@show length(sol.t)


using Plots
plot(sol)




using DiffEqProblemLibrary
DiffEqProblemLibrary.ODEProblemLibrary.importodeproblems()
# prob = DiffEqProblemLibrary.ODEProblemLibrary.prob_ode_brusselator_1d
prob = DiffEqProblemLibrary.ODEProblemLibrary.prob_ode_fitzhughnagumo
solve(prob, ODEFilter(), dt=0.05)
