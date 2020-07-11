# ProbNumODE.jl Documentation

ProbNumODE.jl is a library for [probabilistic numerical methods](http://probabilistic-numerics.org/) for solving ordinary differential equations.
It provides drop-in replacements for classic ODE solvers from [DifferentialEquations.jl](https://docs.sciml.ai/stable/).


## Installation
The package can be installed directly from github:
```julia
] add https://github.com/nathanaelbosch/ProbNumODE.jl
```

## Getting Started
If you are unfamiliar with DifferentialEquations.jl, check out the
[official tutorial](https://docs.sciml.ai/stable/tutorials/ode_example/)
on how to solve ordinary differential equations.
With this in mind, let's set up an `ODEProblem` to solve the
[Fitzhugh-Nagumo model](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model).
```@example 1
using ProbNumODE

function f(u, p, t)
    V, R = u
    a, b, c = p
    return [
        c*(V - V^3/3 + R)
        -(1/c)*(V -  a - b*R)
    ]
end

u0 = [-1.0; 1.0]
tspan = (0., 20.)
p = (0.2,0.2,3.0)
prob = ODEProblem(f, u0, tspan, p)
nothing # hide
```

Currently, ProbNumODE.jl implements two probabilistic numerical methods: `EKF0()` and `EKF1()`.
Both come with additional options, but for now we just use their default values.
```@example 1
sol = solve(prob, EKF0())
@show sol[end]
nothing # hide
```

That's it! You just solved an ODE with a probabilistic numerical method!
Note how the solution object includes uncertainties (using `Measurements.jl`) to describe the numerical approximation error.

Finally, we can visualize the result through [Plots.jl](https://github.com/JuliaPlots/Plots.jl):
```@example 1
using Plots
plot(sol)
savefig("./figures/fitzhugh_nagumo.svg"); nothing # hide
```
![Fitzhugh-Nagumo Solution](./figures/fitzhugh_nagumo.svg?raw=true "Fitzhugh-Nagumo Solution")


## Research References
- Filip Tronarp, Simo Särkkä, Philipp Hennig: **Bayesian Ode Solvers: the Maximum a Posteriori Estimate**
- Filip Tronarp, Hans Kersting, Simo Särkkä, Philipp Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective**
- Michael Schober, Simo Särkkä, Philipp Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems**
- Hans Kersting, T.J. Sullivan, Philipp Hennig: **Convergence Rates of Gaussian Ode Filters**
