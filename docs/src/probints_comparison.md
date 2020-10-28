# Comparison to ProbInts

The DifferentialEquations.jl documentation contains a section about
[Uncertainty Quantification](https://diffeq.sciml.ai/stable/analysis/uncertainty_quantification/).
It describes the
[ProbInts](https://arxiv.org/abs/1506.04592)
method for quantification of numerical uncertainty, and provides an extension of ProbInts to adaptive step sizes.

In this example, we want to compare the uncertainty estimates of `Tsit5`+`AdaptiveProbInts` to the posterior computed with the `EKF1`.


### 1. Problem definition: FitzHugh-Nagumo
```@example probints
using ODEFilters
using ODEFilters: remake_prob_with_jac, stack
using DifferentialEquations
using DiffEqUncertainty
using Statistics
using Plots
pyplot()


function fitz!(du,u,p,t)
    V,R = u
    a,b,c = p
    du[1] = c*(V - V^3/3 + R)
    du[2] = -(1/c)*(V -  a - b*R)
end
u0 = [-1.0;1.0]
tspan = (0.0,20.0)
p = (0.2,0.2,3.0)
prob = ODEProblem(fitz!,u0,tspan,p)
prob = remake_prob_with_jac(prob)
nothing # hide
```

#### High accuracy reference solution:
```@example probints
appxsol = solve(remake(prob, u0=big.(prob.u0)), abstol=1e-20, reltol=1e-20)
plot(appxsol)
savefig("./figures/ex_pi_fitzhugh.svg"); nothing # hide
```
![Prob-Ints Errors](./figures/ex_pi_fitzhugh.svg)


### 2. ProbInts
Uncertainty quantification of `Tsit5` with `AdaptiveProbInts`:
```@example probints
cb = AdaptiveProbIntsUncertainty(5)
ensemble_prob = EnsembleProblem(prob)
sol = solve(prob, Tsit5())
sim = solve(ensemble_prob, Tsit5(), trajectories=100, callback=cb)

p = plot(sol.t, stack(appxsol.(sol.t) - sol.u), color=[3 4], ylims=(-0.003, 0.003), ylabel="Error")
errors = [(a.t, stack(appxsol.(a.t) .- a.u)) for a in sim.u]
for e in errors
    plot!(p, e[1], e[2], color=[3 4], label="", linewidth=0.2, linealpha=0.5)
end
savefig("./figures/ex_pi_probints.svg"); nothing # hide
```
![Prob-Ints Errors](./figures/ex_pi_probints.svg)


### 3. EKF1
Uncertainties provided by the `EKF1`:
```@example probints
sol = solve(prob, EKF1())
plot(sol.t, stack(appxsol.(sol.t) - sol.u), ylabel="Error")
plot!(sol.t, zero(stack(sol.u)), ribbon=3stack(std(sol.pu)), color=[1 2], label="")
savefig("./figures/ex_pi_ours.svg"); nothing # hide
```
![Our Errors](./figures/ex_pi_ours.svg)

Verdict: The provided credible bands are more calibrated!
