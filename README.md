# ODEFilters.jl

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/ODEFilters.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/ODEFilters.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/ODEFilters.jl/workflows/CI/badge.svg)](https://github.com/nathanaelbosch/ODEFilters.jl/actions)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ODEFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nathanaelbosch/ODEFilters.jl)
<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->
<!-- [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) -->


**ODEFilters.jl provides _probabilistic_ ODE solvers for the [DifferentialEquations.jl](https://docs.sciml.ai/stable/) ecosystem.**
By casting the solution of ODEs as a problem of Bayesian inference, they return a distribution over ODE solutions and provide estimates of their own numerical approximation error.

<!-- [Probabilistic Numerics](http://probabilistic-numerics.org/). -->

![Fitzhugh-Nagumo Solve Animation](./examples/fitzhughnagumo_solve.gif?raw=true "Fitzhugh-Nagumo Solve Animation")


## Installation
The package can be installed directly from github:
```julia
] add https://github.com/nathanaelbosch/ODEFilters.jl
```


## Example
Solving ODEs with probabilistic numerical methods is as simple as that!
```julia
using ODEFilters

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

# Solve the ODE with a probabilistic numerical solver: EKF0
sol = solve(prob, EKF0(), abstol=1e-1, reltol=1e-2)

# Plot the solution
using Plots
plot(sol)
```
![Fitzhugh-Nagumo Solution](./docs/src/figures/fitzhugh_nagumo.svg?raw=true "Fitzhugh-Nagumo Solution")
