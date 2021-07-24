# Getting Started
If you are unfamiliar with DifferentialEquations.jl, check out the
[official tutorial](https://docs.sciml.ai/stable/tutorials/ode_example/)
on how to solve ordinary differential equations.


## Step 1: Defining a problem
To solve the
[Fitzhugh-Nagumo model](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model)
we first set up an `ODEProblem`.
```@example 1
using ProbNumDiffEq

function fitz(u, p, t)
    a, b, c = p
    return [c*(u[1] - u[1]^3/3 + u[2])
            -(1/c)*(u[1] -  a - b*u[2])]
end

u0 = [-1.0; 1.0]
tspan = (0., 20.)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(fitz, u0, tspan, p)
nothing # hide
```

## Step 2: Solving a problem
To solve the `ODEProblem` we can use the `solve` interface that DifferentialEquations.jl defines.
All we have to do is to select one of the PN algorithms: `EK0` or `EK1`.
In this example we solve the ODE with the default `EK0` and high tolerance levels to visualize the resulting uncertainty
```@example 1
sol = solve(prob, EK0(), abstol=1e-1, reltol=1e-1)
nothing # hide
```

Note that ProbNumDiffEq.jl supports many of DifferentialEquations.jl's
[common solver options](https://diffeq.sciml.ai/stable/basics/common_solver_opts/).


## Step 3: Analyzing the solution
Just as in DifferentialEquations.jl, the result of `solve` is a solution object, and we can access the (mean) values and timesteps as usual
```@repl 1
sol[end]
sol.u[5]
sol.t[8]
```

However, the solver returns a _probabilistic_ solution, here a
[Gaussian](https://github.com/mschauer/GaussianDistributions.jl)
distribution over solution values:
```@repl 1
sol.pu[end]
```

It is often convenient to look at means, covariances, and standard deviations via Statistics.jl:
```@repl 1
using Statistics
mean(sol.pu[5])
cov(sol.pu[5])
std(sol.pu[5])
```

By default, the posterior distribution can be evaluated for arbitrary points in time `t` by treating `sol` as a function:
```@repl 1
mean(sol(0.45))
```

### Plotting
The result can be conveniently visualized through [Plots.jl](https://github.com/JuliaPlots/Plots.jl):
```@example 1
using Plots
plot(sol, color=["#107D79" "#FF9933"])
savefig("./figures/fitzhugh_nagumo.svg"); nothing # hide
```
![Fitzhugh-Nagumo Solution](./figures/fitzhugh_nagumo.svg)
