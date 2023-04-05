# ProbNumDiffEq.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/workflows/CI/badge.svg)](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/actions)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ProbNumDiffEq.jl/branch/main/graph/badge.svg?token=eufIemCGXn)](https://codecov.io/gh/nathanaelbosch/ProbNumDiffEq.jl)
[![Benchmarks](http://img.shields.io/badge/benchmarks-docs-blueviolet.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/dev/benchmarks/multi-language-wrappers/)

<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->
<!-- [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) -->
![Banner](./examples/banner.svg?raw=true)

__ProbNumDiffEq.jl__ provides _probabilistic numerical_ ODE solvers to the
[DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) ecosystem.
The implemented _ODE filters_ solve differential equations via Bayesian filtering and smoothing. The filters compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.

For a short intro video, check out our [poster presentation at JuliaCon2021](https://www.youtube.com/watch?v=EMFl6ytP3iQ).


## Installation

Run Julia, enter `]` to bring up Julia's package manager, and add the ProbNumDiffEq.jl package:

```
julia> ]
(v1.8) pkg> add ProbNumDiffEq
```


## Example: Solving the FitzHugh-Nagumo ODE

```julia
using ProbNumDiffEq

# ODE definition as in DifferentialEquations.jl
function f(du, u, p, t)
    a, b, c = p
    du[1] = c * (u[1] - u[1]^3 / 3 + u[2])
    du[2] = -(1 / c) * (u[1] - a - b * u[2])
end
u0 = [-1.0, 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(f, u0, tspan, p)

# Solve the ODE with a probabilistic numerical solver: EK1
sol = solve(prob, EK1())

# Plot the solution with Plots.jl
using Plots
plot(sol, color=["#CB3C33" "#389826" "#9558B2"])
```

![Fitzhugh-Nagumo Solution](./examples/fitzhughnagumo.svg?raw=true "Fitzhugh-Nagumo Solution")

Since we're doing probabilistic numerics the solution also contains error estimates, it just happens that they are too small to be visible in the plot above.
But we can just plot them directly:

```julia
using Statistics
stds = std.(sol.pu)
plot(sol.t, hcat(stds...)', color=["#CB3C33" "#389826" "#9558B2"]
     label=["std(u1(t))" "std(u2(t))"], xlabel="t", ylabel="standard-deviation")
```

![Fitzhugh-Nagumo Standard-Deviations](./examples/fitzhughnagumo_stddevs.svg?raw=true "Fitzhugh-Nagumo Standard-Deviations")


## Related packages

- [probdiffeq](https://pnkraemer.github.io/probdiffeq/): Fast and feature-rich filtering-based probabilistic ODE solvers in JAX.
- [ProbNum](https://probnum.readthedocs.io/en/latest/): Probabilistic numerics in Python. It has not only probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.
- [Fenrir.jl](https://github.com/nathanaelbosch/Fenrir.jl): Parameter-inference in ODEs with probabilistic ODE solvers. This package builds on ProbNumDiffEq.jl to provide a negative marginal log-likelihood function, which can then be used with an optimizer or with MCMC for parameter inference.


## References

The main references _for this package_ include:

  - M. Schober, S. Särkkä, and P. Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems** (2018) ([link](https://link.springer.com/article/10.1007/s11222-017-9798-7))
  - F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019) ([link](https://link.springer.com/article/10.1007/s11222-019-09900-1))
  - N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020) ([link](https://arxiv.org/abs/2012.10106))
  - N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021) ([link](http://proceedings.mlr.press/v130/bosch21a.html))
  - N. Bosch, F. Tronarp, P. Hennig: **Pick-and-Mix Information Operators for Probabilistic ODE Solvers** (2022) ([link](https://proceedings.mlr.press/v151/bosch22a.html))

A more extensive list of references relevant to ODE filters is provided [here](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable/#References).
