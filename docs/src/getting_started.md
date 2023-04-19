# Solving ODEs with Probabilistic Numerics

In this tutorial we solve a simple non-linear ordinary differential equation (ODE) with the _probabilistic numerical_ ODE solvers implemented in this package.

!!! note
    If you never used DifferentialEquations.jl, check out their
    ["Getting Started with Differential Equations in Julia" tutorial](https://docs.sciml.ai/DiffEqDocs/stable/getting_started/).
    It explains how to define and solve ODE problems and how to analyze the solution, so it's a great starting point.
    Most of ProbNumDiffEq.jl works exaclty as you would expect from DifferentialEquations.jl -- just with some added uncertainties and related functionality on top!


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

## TL;DR: Just use DifferentialEquations.jl with the `EK1` algorithm

```@example 1
using ProbNumDiffEq, Plots

function fitz(du, u, p, t)
    a, b, c = p
    du[1] = c * (u[1] - u[1]^3 / 3 + u[2])
    du[2] = -(1 / c) * (u[1] - a - b * u[2])
end
u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(fitz, u0, tspan, p)

using Logging; Logging.disable_logging(Logging.Warn); # hide
sol = solve(prob, EK1())
Logging.disable_logging(Logging.Debug) # hide
plot(sol)
```

## Step 1: Define the problem

First, import ProbNumDiffEq.jl

```@example 1
using ProbNumDiffEq
```

Then, set up the `ODEProblem` exactly as you would in DifferentialEquations.jl.
Define the vector field

```@example 1
function fitz(du, u, p, t)
    a, b, c = p
    du[1] = c * (u[1] - u[1]^3 / 3 + u[2])
    du[2] = -(1 / c) * (u[1] - a - b * u[2])
end
nothing # hide
```

and then the `ODEProblem`, with initial value `u0`, time span `tspan`, and parameters `p`

```@example 1
u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(fitz, u0, tspan, p)
nothing # hide
```

## Step 2: Solve the problem

To solve the ODE we just use DifferentialEquations.jl's `solve` interface, together with one of the algorithms implemented in this package.
For now, let's use [`EK1`](@ref):

```@example 1
sol = solve(prob, EK1())
# nothing # hide
```

That's it! we just computed a _probabilistic numerical_ ODE solution!


## Step 3: Analyze the solution

Let's plot the result with [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

```@example 1
using Plots
plot(sol)
```

Looks good! Looks like the `EK1` managed to solve the Fitzhugh-Nagumo problem quite well.


!!! tip
    To learn more about plotting ODE solutions, check out the plotting tutorial for DifferentialEquations.jl + Plots.jl provided [here](https://docs.sciml.ai/DiffEqDocs/stable/basics/plot/).
    Most of that works exactly as expected with ProbNumDiffEq.jl.


### Plot the probabilistic error estimates

The plot above looks like a standard ODE solution -- but it's not!
The numerical errors are just so small that we can't see them in the plot, and the probabilistic error estimates are too.
We can visualize them by plotting the errors and error estimates directly:

```@example 1
using OrdinaryDiffEq, Statistics
reference = solve(prob, Vern9(), abstol=1e-9, reltol=1e-9, saveat=sol.t)
errors = reduce(hcat, mean.(sol.pu) .- reference.u)'
error_estimates = reduce(hcat, std.(sol.pu))'
plot(sol.t, errors, label="error", color=[1 2], xlabel="t", ylabel="err")
plot!(sol.t, zero(errors), ribbon=3error_estimates, label="error estimate",
      color=[1 2], alpha=0.2)
```

### More about the `ProbabilisticODESolution`

The solution object returned by ProbNumDiffEq.jl mostly behaves just like any other ODESolution in DifferentialEquations.jl --
with some added uncertainties and related functionality on top.
So, `sol` can be indexed

```@repl 1
sol[1]
sol[end]
```

and has fields `sol.t` and `sol.u` which store the time points and mean estimates:

```@repl 1
sol.t[end]
sol.u[end]
```


But since `sol` is a _probabilistic numerical_ ODE solution, it contains a
[Gaussian](https://github.com/mschauer/GaussianDistributions.jl)
distributions over solution values.
The marginals of this posterior are stored in `sol.pu`:

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

#### Dense output

Probabilistic numerical ODE solvers approximate the posterior distribution

```math
p \Big( y(t) ~\big|~ y(0) = y_0, \{ \dot{y}(t_i) = f_\theta(y(t_i), t_i) \} \Big),
```

which describes a posterior not just for the discrete steps but for any ``t`` in the continuous space ``t \in [0, T]``;
in classic ODE solvers, this is also known as "interpolation" or "dense output".
The probabilistic solutions returned by our solvers can be interpolated as usual by treating them as functions,
but they return Gaussian distributions

```@repl 1
sol(0.45)
mean(sol(0.45))
```


## Next steps

Check out one of the other tutorials:
- "[Second Order ODEs and Energy Preservation](@ref)" explains how to solve second-order ODEs more efficiently while also better perserving energy or other conserved quantities;
- "[Solving DAEs with Probabilistic Numerics](@ref)" demonstrates how to solve differential algebraic equatios in a probabilistic numerical way.
