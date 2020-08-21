## Uncertainty calibration: Ours vs probints
Comparison with the ProbInts approach as implemented in DiffEqUncertainty.

**THIS COMPARISON IS STILL WIP!**

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
sol = solve(prob, EKF0(), q=3, steprule=:constant, dt=dt)
#sol = solve(prob, EKF0(), q=3, abstol=abstol, reltol=reltol)
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
@benchmark sol = solve(prob, EKF0(), q=3, steprule=:constant, dt=dt)
@benchmark sim = solve(ensemble_prob, Tsit5(), adaptive=false, dt=dt, trajectories=100, callback=cb)
```
