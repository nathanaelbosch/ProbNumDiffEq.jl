# ProbNumDiffEq.jl



[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/workflows/CI/badge.svg)](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/actions)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ProbNumDiffEq.jl/branch/main/graph/badge.svg?token=eufIemCGXn)](https://codecov.io/gh/nathanaelbosch/ProbNumDiffEq.jl)
[![Benchmarks](http://img.shields.io/badge/benchmarks-ipynb-blueviolet.svg)](./benchmarks/)
<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->
<!-- [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) -->


![Banner](./examples/banner.svg?raw=true)

__ProbNumDiffEq.jl__ provides _probabilistic numerical_ ODE solvers to the
[DifferentialEquations.jl](https://docs.sciml.ai/stable/) ecosystem.
The implemented _ODE filters_ solve differential equations via Bayesian filtering and smoothing and compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.


---

__For more probabilistic numerics check out the [ProbNum](https://www.probabilistic-numerics.org/en/latest/) Python package.__
It implements probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.

---

## Installation
The package can be installed directly with the Julia package manager:
```julia
] add ProbNumDiffEq
```


## Example: Solving the FitzHugh-Nagumo ODE
```julia
using ProbNumDiffEq

# ODE definition as in DifferentialEquations.jl
function fitz(u, p, t)
    a, b, c = p
    return [c*(u[1] - u[1]^3/3 + u[2])
            -(1/c)*(u[1] -  a - b*u[2])]
end
u0 = [-1.0; 1.0]
tspan = (0., 20.)
p = (0.2,0.2,3.0)
prob = ODEProblem(fitz, u0, tspan, p)

# Solve the ODE with a probabilistic numerical solver: EK0
sol = solve(prob, EK0(order=1), abstol=1e-1, reltol=1e-2)

# Plot the solution
using Plots
C1, C2 = "#107D79", "#FF9933"
plot(sol, color=[C1 C2], fillalpha=0.15)

# Sample from the solution and plot the samples
samples = ProbNumDiffEq.sample(sol, 100)
for i in 1:100
    plot!(sol.t, samples[:, :, i], color=[C1 C2], label="", linewidth=0.1)
end
```
![Fitzhugh-Nagumo Solution](./docs/src/figures/fitzhugh_nagumo.svg?raw=true "Fitzhugh-Nagumo Solution")


## Benchmarks

- [Multi-Language Wrapper Benchmark](./benchmarks/multi-language-wrappers.ipynb):
  ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.
