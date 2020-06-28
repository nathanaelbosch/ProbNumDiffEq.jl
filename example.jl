using DiffEqFilters


# Current code
prob = fitzhugh_nagumo()
psol = prob_solve(prob, 0.05);
@show length(psol.t)


sol = solve(prob, ODEFilter(), dt=0.05)
@show length(sol.t)


using Plots
plot(sol)
