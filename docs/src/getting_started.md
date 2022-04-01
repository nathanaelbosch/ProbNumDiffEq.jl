# Solving ODEs with Probabilistic Numerics
In this tutorial we solve a simple non-linear ordinary differential equation (ODE) with the _probabilistic numerical_ ODE solvers implemented in this package.

!!! note
    If you never used DifferentialEquations.jl, check out their
    ["Ordinary Differential Equations" tutorial](https://diffeq.sciml.ai/stable/tutorials/ode_example/)
    on how to solve ordinary differential equations with classic numerical solvers.

In this tutorial, we consider a
[Fitzhugh-Nagumo model](https://en.wikipedia.org/wiki/FitzHugh%E2%80%93Nagumo_model)
described by an ODE of the form
```math
\begin{aligned}
\dot{y}_1 &= c (y_1 - \frac{y_1^3}{3} + y_2) \\
\dot{y}_2 &= -\frac{1}{c} (y_1 - a - b y_2)
\end{aligned}
```
on a time span ``t \in [0, T]``, with initial value ``y(0) = y_0``.
In the following, we
1. define the problem with explicit choices of initial values, integration domains, and parameters,
2. solve the problem with our ODE filters, and
3. visualize the results and the corresponding uncertainties.


#### TL;DR:
```@example 1
using ProbNumDiffEq, Plots

function fitz(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1/c)*(u[1] -  a - b*u[2])
end
u0 = [-1.0; 1.0]
tspan = (0., 20.)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(fitz, u0, tspan, p)

using Logging; Logging.disable_logging(Logging.Warn) # hide
sol = solve(prob, EK1())
Logging.disable_logging(Logging.Debug) # hide
plot(sol)
```


## Step 1: Defining the problem
We first import ProbNumDiffEq.jl
```@example 1
using ProbNumDiffEq
```
and then set up an `ODEProblem` exactly as we're used to with DifferentialEquations.jl,
by defining the vector field
```@example 1
function fitz(du, u, p, t)
    a, b, c = p
    du[1] = c*(u[1] - u[1]^3/3 + u[2])
    du[2] = -(1/c)*(u[1] -  a - b*u[2])
end
nothing # hide
```
and then an `ODEProblem`, with initial value `u0`, time span `tspan`, and parameters `p`
```@example 1
u0 = [-1.0; 1.0]
tspan = (0., 20.)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(fitz, u0, tspan, p)
nothing # hide
```

## Step 2: Solving the problem
To solve the ODE we just use DifferentialEquations.jl's `solve` interface, together with one of the algorithms implemented in this package.
For now, let's use `EK1`:
```@example 1
using Logging; Logging.disable_logging(Logging.Warn) # hide
sol = solve(prob, EK1())
Logging.disable_logging(Logging.Debug) # hide
nothing # hide
```
That's it! we just computed a _probabilistic numerical_ ODE solution!


## Step 3: Analyzing the solution
The result of `solve` is a solution object which can be handled just as in DifferentialEquations.jl.
We can access _mean_ values by indexing `sol`
```@repl 1
sol[end]
```
or directly via `sol.u`
```@repl 1
sol.u[end]
```
and similarly the time steps
```@repl 1
sol.t[end]
```

But we didn't use probabilstic numerics to just compute means.
In fact, `sol` is a _probabilistic numerical_ ODE solution and it provides
[Gaussian](https://github.com/mschauer/GaussianDistributions.jl)
distributions over solution values.
These are stored in `sol.pu`:
```@repl 1
sol.pu[end]
```

You can compute means, covariances, and standard deviations via Statistics.jl:
```@repl 1
using Statistics
mean(sol.pu[5])
cov(sol.pu[5])
std(sol.pu[5])
```

### Dense output
Probabilistic numerical ODE solvers approximate the posterior distribution
```math
p \Big( y(t) \mid \{ \dot{y}(t_i) = f_\theta(y(t_i), t_i) \} \Big),
```
which describes a posterior not just for the discrete steps but for any ``t`` in the continuous space ``t \in [0, T]``;
in classic ODE solvers, this is also known as "interpolation" or "dense output".
The probabilistic solutions returned by our solvers can be interpolated as usual by treating them as functions,
but they return Gaussian distributions
```@repl 1
sol(0.45)
mean(sol(0.45))
```

### Plotting
The result can be conveniently visualized through [Plots.jl](https://github.com/JuliaPlots/Plots.jl):
```@example 1
using Plots
plot(sol)
```

A more detailed plotting tutorial for DifferentialEquations.jl + Plots.jl is provided [here](https://diffeq.sciml.ai/stable/basics/plot/); most of the features work exactly as expected.


The uncertainties here are very low compared to the function value so we can't really see them.
Just to demonstrate that they're there, let's solve the explicit `EK0` solver, low order and higher tolerance levels:
```@example 1
using Logging; Logging.disable_logging(Logging.Warn) # hide
sol = solve(prob, EK0(order=1), abstol=1e-2, reltol=1e-1)
Logging.disable_logging(Logging.Debug) # hide
plot(sol, denseplot=false)
```

There it is!
