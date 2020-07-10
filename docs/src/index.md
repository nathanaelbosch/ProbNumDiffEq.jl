# ProbNumODE.jl Documentation

ProbNumODE.jl is a library for probabilistic numerical methods for solving ordinary differential equations.
It provides drop-in replacements for classic ODE solvers from [DifferentialEquations.jl](https://docs.sciml.ai/stable/).

## Minimal Example
Solving ODEs probabilistically is as simple as that:
```@example 1
using ProbNumODE

prob = fitzhugh_nagumo()
sol = solve(prob, EKF1())
nothing # hide
```

Visualizations are available through [Plots.jl](https://github.com/JuliaPlots/Plots.jl):
```@example 1
using Plots
plot(sol)
# mkdir("./figures") # hide
savefig("./figures/fitzhugh_nagumo.svg"); nothing # hide
```
![](./figures/fitzhugh_nagumo.svg)


## Research References
- Filip Tronarp, Simo Särkkä, Philipp Hennig: **Bayesian Ode Solvers: the Maximum a Posteriori Estimate**
- Filip Tronarp, Hans Kersting, Simo Särkkä, Philipp Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective**
- Michael Schober, Simo Särkkä, Philipp Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems**
- Hans Kersting, T.J. Sullivan, Philipp Hennig: **Convergence Rates of Gaussian Ode Filters**
