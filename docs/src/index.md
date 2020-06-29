# ProbNumODE.jl Documentation


## Minimal Example
Solving ODEs probabilistically is as simple as that:
```julia
using ProbNumODE

prob = fitzhugh_nagumo()
sol = solve(prob, ODEFilter())
```


## Plotting Functionality
Requires [Plots.jl](https://github.com/JuliaPlots/Plots.jl).
```julia
using Plots
plot(sol)
```
![Fitzhugh-Nagumo Solution](./figures/fitzhugh_nagumo.png?raw=true "Fitzhugh-Nagumo Solution")


## Compatible with problems from DiffEqProblemLibrary.jl
Requires [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl)
```julia
using DiffEqProblemLibrary
DiffEqProblemLibrary.ODEProblemLibrary.importodeproblems()

prob = DiffEqProblemLibrary.ODEProblemLibrary.prob_ode_rigidbody
sol = solve(prob, ODEFilter())
plot(sol)
```
![Rigid Body Equations](./figures/rigidbody.png?raw=true "Rigid Body Equations Solution")


## Symbolic problem definition
Requires [DifferentialEquations.jl](https://docs.sciml.ai/stable/)
```julia
using DifferentialEquations
using ProbNumODE

f = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d
p = [1.5,1.0,3.0,1.0]
u0 = [1.0,1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob, ODEFilter())
plot(sol)
```
![Lotka-Volterra Solution](./figures/lotka-volterra.png?raw=true "Lotka-Volterra Solution**)


## Uncertain starting values
**Does this even make sense in our method??**
I see at least one problem: The adaptive step size tries to limit uncertainty in the output.
If the starting value is uncertain, it seems unreasonable to provide a tolerance level for the output uncertainty.

Comparison of different methods:
1. Ours
2. DifferentialEquations.jl + Measurements.jl
3. DifferentialEquations.jl + sampling

```julia
using DifferentialEquations
using ProbNumODE

f = @ode_def begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a b c d
p = [1.5,1.0,3.0,1.0]
u0 = [1.0,1.0]
tspan = (0.0,10.0)
base_prob = ODEProblem(f,u0,tspan,p)

# 1. Ours
prob = ODEProblem(f,u0 .± 0.1,tspan,p)
sol = solve(prob, ODEFilter(), steprule=:constant, q=3, dt=0.01)

# 2. Reference: Classic solver + Measurements.jl
sol_classic = solve(prob, abstol=1e-16, reltol=1e-16)
# Plotting
ts = range(sol_classic.t[1], sol_classic.t[end], length=1000)
us = sol(trange)
values = [Measurements.value.(u) for u in us]
uncertainties = [Measurements.uncertainty.(u) for u in us]


# 3. Reference: Classic solver + Sampling
function add_init_noise(prob, i, repeat; noise=0.1)
    remake(prob,u0=prob.u0 .+ noise * randn(size(prob.u0)))
end
ensemble_solution = solve(EnsembleProblem(base_prob, prob_func=add_init_noise),
                          trajectories=100, abstol=1e-16, reltol=1e-16)
# p3 = plot(ensemble_solution)


plot(sol, color=1, fillalpha=0.3, width=1, label="PN")
plot!(ts, stack(values), ribbon=1.96*stack(uncertainties),
      color=2, fillalpha=0.1, width=1, label="Measurements.jl")
plot!(EnsembleSummary(ensemble_solution),
      color=3, fillalpha=0.1, label="Sampling", width=1)
```
![Uncertain Init Comparison](./figures/uncertain_init_comparison.png?raw=true "Uncertain Initial Value Comparison")
