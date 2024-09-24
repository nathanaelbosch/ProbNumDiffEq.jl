# ProbNumDiffEq.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/stable)
[![Development](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/workflows/CI/badge.svg)](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/actions)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ProbNumDiffEq.jl/branch/main/graph/badge.svg?token=eufIemCGXn)](https://codecov.io/gh/nathanaelbosch/ProbNumDiffEq.jl)
[![Benchmarks](http://img.shields.io/badge/benchmarks-docs-blueviolet.svg)](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/dev/benchmarks/multi-language-wrappers/)

![Banner](./examples/banner.svg?raw=true)

__ProbNumDiffEq.jl__ provides _probabilistic numerical_ ODE solvers to the
[DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) ecosystem.
The implemented _ODE filters_ solve differential equations via Bayesian filtering and smoothing. The filters compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.

For a short intro video, check out the [ProbNumDiffEq.jl poster presentation at JuliaCon2021](https://www.youtube.com/watch?v=EMFl6ytP3iQ).


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

In probabilistic numerics, the solution also contains error estimates - it just happens that they are too small to be visible in the plot above.
But we can just plot them directly:

```julia
using Statistics
stds = std.(sol.pu)
plot(sol.t, hcat(stds...)', color=["#CB3C33" "#389826" "#9558B2"],
     label=["std(u1(t))" "std(u2(t))"], xlabel="t", ylabel="standard-deviation")
```

![Fitzhugh-Nagumo Standard-Deviations](./examples/fitzhughnagumo_stddevs.svg?raw=true "Fitzhugh-Nagumo Standard-Deviations")


## Contributing

**Contributions are very welcome!**
Check the existing issues for ideas on how to contribute to the package. 
If you want to implement a new functionality/algorithm, open an issue to start a discussion.

**Please open issues liberally!**
If there is anything that's unclear or doesn't work, we would very much like to know about it.
This includes not just bugs and feature requests but also general questions about the software, feedback and suggestions.


## Related packages

- [ProbDiffEq](https://pnkraemer.github.io/probdiffeq/) is similar in scope to ProbNumDiffEq.jl and it provides fast and feature-rich probabilistic ODE solvers but is implemented in Python and built on JAX.
- [ProbNum](https://probnum.readthedocs.io/en/latest/) implements a wide range of probabilistic numerical methods, not only for ODEs but also for linear algebra, quadrature, and filtering/smoothing. It is implemented in Python and NumPy, and it focuses more on breadth and didactic purposes than on performance.
