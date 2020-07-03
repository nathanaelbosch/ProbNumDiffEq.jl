# ProbNumODE.jl Documentation


## Minimal Example
Solving ODEs probabilistically is as simple as that:
```@example 1
using ProbNumODE

prob = fitzhugh_nagumo()
sol = solve(prob, ODEFilter())
nothing # hide
```


## Plotting Functionality
Requires [Plots.jl](https://github.com/JuliaPlots/Plots.jl).
```@example 1
using Plots
plot(sol)
mkdir("./figures") # hide
savefig("./figures/fitzhugh_nagumo.svg"); nothing # hide
```
![](./figures/fitzhugh_nagumo.svg)


## Compatible with problems from DiffEqProblemLibrary.jl
Requires [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl)
```@example
using ProbNumODE
using Plots
using DiffEqProblemLibrary
DiffEqProblemLibrary.ODEProblemLibrary.importodeproblems()

prob = DiffEqProblemLibrary.ODEProblemLibrary.prob_ode_rigidbody
sol = solve(prob, ODEFilter())
plot(sol)
savefig("./figures/rigidbody.svg"); nothing # hide
```
![](./figures/rigidbody.svg)


## Symbolic problem definition
Requires [DifferentialEquations.jl](https://docs.sciml.ai/stable/)
```@example
using DifferentialEquations
using ProbNumODE
using Plots

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
savefig("./figures/lotka-volterra.svg"); nothing # hide
```
![](./figures/lotka-volterra.svg)


## Uncertain starting values
**Does this even make sense in our method??**
I see at least one problem: The adaptive step size tries to limit uncertainty in the output.
If the starting value is uncertain, it seems unreasonable to provide a tolerance level for the output uncertainty.

Comparison of different methods:
1. Ours
2. DifferentialEquations.jl + Measurements.jl
3. DifferentialEquations.jl + sampling

```@example
using DifferentialEquations
using ProbNumODE
using Plots
using Measurements

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
us = sol_classic(ts)
values = [Measurements.value.(u) for u in us]
uncertainties = [Measurements.uncertainty.(u) for u in us]


# 3. Reference: Classic solver + Sampling
function add_init_noise(prob, i, repeat; noise=0.1)
    remake(prob,u0=prob.u0 .+ noise * randn(size(prob.u0)))
end
ensemble_solution = solve(EnsembleProblem(base_prob, prob_func=add_init_noise),
                          trajectories=100, abstol=1e-10, reltol=1e-10)
# p3 = plot(ensemble_solution)


plot(sol, color=1, fillalpha=0.3, width=1, label="PN")
stack(x) = copy(reduce(hcat, x)')
plot!(ts, stack(values), ribbon=1.96*stack(uncertainties),
      color=2, fillalpha=0.1, width=1, label="Measurements.jl")
plot!(EnsembleSummary(ensemble_solution),
      color=3, fillalpha=0.1, label="Sampling", width=1)
savefig("./figures/uncertain_init_comparison.svg"); nothing # hide
```
![](./figures/uncertain_init_comparison.svg)


## Uncertainty calibration: Ours vs probints
Comparison with the ProbInts approach as implemented in DiffEqUncertainty.
```@example probints
using DifferentialEquations
using ProbNumODE
using Plots
using BenchmarkTools
using DiffEqUncertainty
using LinearAlgebra
using Measurements
using Distributions

function f(u, p, t)
    a, b, c, d = p
    return [a*u[1] - b*u[1]*u[2];
            -c*u[2] + d*u[1]*u[2]]
end
p = [1.5,1.0,3.0,1.0]
u0 = [1.0,1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan,p)

#1. Ours
dt = 0.01
abstol, reltol = 1e-2, 1e-2
sol = solve(prob, ODEFilter(), q=3, steprule=:constant, dt=dt)
#sol = solve(prob, ODEFilter(), q=3, abstol=abstol, reltol=reltol)
plot(sol)

cb = AdaptiveProbIntsUncertainty(5)
ensemble_prob = EnsembleProblem(prob)
sim = solve(ensemble_prob, Tsit5(), adaptive=false, dt=dt, trajectories=100, callback=cb)
#sim = solve(ensemble_prob, Tsit5(), trajectories=100, callback=cb, abstol=abstol, reltol=reltol)
plot!(EnsembleSummary(sim))


best = solve(prob, abstol=1e-16, reltol=1e-16)

#Our errors
means = [Measurements.value.(u) for u in sol.u]
vars = [Measurements.uncertainty.(u) .^ 2 for u in sol.u]
diffs = best(sol.t).u .- means
cal = [d' * Diagonal(1 ./ v) * d for (d, v) in zip(diffs, vars)]
stack(x) = copy(reduce(hcat, x)')
plot(sol.t[2:end], stack(cal[2:end]), yscale=:log10, label="Ours")

#Their errors
es = EnsembleSummary(sim)
diffs2 = best(es.t).u .- es.u.u
cal2 = [d' * Diagonal(1 ./ v) * d for (d, v) in zip(diffs2, es.v.u)]
plot!(es.t[3:end], stack(cal2[3:end]), label="Classic+ProbInts")

p = 1-0.95
hline!([quantile(Chisq(sol.solver.d), 1-(p/2)),
        quantile(Chisq(sol.solver.d), p/2)],
seriescolor=:black,
            linestyle=:dash,
            label="",
)
hline!([quantile(Chisq(sol.solver.d), 0.5)], seriescolor=:black, label="")
savefig("./figures/uncertainty_calibration.svg"); nothing # hide
```
![](./figures/uncertainty_calibration.svg)
**Note:** This comparison might not yet be really fair. I think I need to allow adaptive step sizes, since that's what `AdaptiveProbIntsUncertainty** is made for and then it supposedly uses some method for uncertainty calibration, which might not really be the case here.

**Bonus:**Benchmarking
```@repl probints
@benchmark sol = solve(prob, ODEFilter(), q=3, steprule=:constant, dt=dt)
@benchmark sim = solve(ensemble_prob, Tsit5(), adaptive=false, dt=dt, trajectories=100, callback=cb)
```
