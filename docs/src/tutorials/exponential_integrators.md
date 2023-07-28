# Probabilistic Exponential Integrators

Exponential integrators are a class of numerical methods for solving semi-linear ordinary differential equations (ODEs) of the form
```math
\begin{aligned}
\dot{y}(t) &= L y(t) + f(y(t), t), \quad y(0) = y_0,
\end{aligned}
```
where $L$ is a linear operator and $f$ is a nonlinear function.
In a nutshell, exponential integrators solve the linear part of the ODE exactly, and only approximate the nonlinear part.
[Probabilistic exponential integrators](@cite bosch23expint) [bosch23expint](@cite) are the probabilistic numerics approach to exponential integrators.

## Example

Let's consider a simple semi-linear ODE
```math
\begin{aligned}
\dot{y}(t) &= - y(t) + \sin(y(t)), \quad y(0) = 1.0.
\end{aligned}
```

We can solve this ODE reasonably well with the standard [`EK1`](@ref)  and adaptive steps (the default):
```@example expint
using ProbNumDiffEq, Plots, LinearAlgebra
theme(:default; palette=["#4063D8", "#389826", "#9558B2", "#CB3C33"])

f(du, u, p, t) = (@. du = -u + sin(u))
u0 = [1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(f, u0, tspan)

ref = solve(prob, EK1(), abstol=1e-10, reltol=1e-10)
plot(ref, color=:black, linestyle=:dash, label="Reference")
```

But for fixed (large) step sizes this ODE is more challenging:
The explicit [`EK0`](@ref) method oscillates and diverges due to the stiffness of the ODE,
and the semi-implicit [`EK1`](@ref) method is stable but the solution is not very accurate.
```@example expint
STEPSIZE = 4
DM = FixedDiffusion() # recommended for fixed steps

# we don't smooth the EK0 here to show the oscillations more clearly
sol0 = solve(prob, EK0(smooth=false, diffusionmodel=DM), adaptive=false, dt=STEPSIZE, dense=false)
sol1 = solve(prob, EK1(diffusionmodel=DM), adaptive=false, dt=STEPSIZE)

plot(ylims=(0.3, 1.05))
plot!(ref, color=:black, linestyle=:dash, label="Reference")
plot!(sol0, denseplot=false, marker=:o, markersize=2, label="EK0", color=1)
plot!(sol1, denseplot=false, marker=:o, markersize=2, label="EK1", color=2)
```

_**Probabilistic exponential integrators**_ leverage the semi-linearity of the ODE to compute more accurate solutions for the same fixed step size.
You can use either the [`ExpEK`](@ref) method and provide the linear part (with the keyword argument `L`),
or the [`RosenbrockExpEK`](@ref) to automatically linearize along the mean of the numerical solution:
```@example expint
sol_exp = solve(prob, ExpEK(L=-1, diffusionmodel=DM), adaptive=false, dt=STEPSIZE)
sol_ros = solve(prob, RosenbrockExpEK(diffusionmodel=DM), adaptive=false, dt=STEPSIZE)

plot(ylims=(0.3, 1.05))
plot!(ref, color=:black, linestyle=:dash, label="Reference")
plot!(sol_exp, denseplot=false, marker=:o, markersize=2, label="ExpEK", color=3)
plot!(sol_ros, denseplot=false, marker=:o, markersize=2, label="RosenbrockExpEK", color=4)
```

The solutions are indeed much more accurate than those of the standard `EK1`, for the same fixed step size!


## Background: Integrated Ornstein-Uhlenbeck priors

Probabilistic exponential integrators "solve the linear part exactly" by including it into the prior model of the solver.
Namely, the solver chooses a (q-times) integrated Ornstein-Uhlenbeck prior with rate parameter equal to the linearity.
The [`ExpEK`](@ref) solver is just a short-hand for an [`EK0`](@ref) with appropriate prior:
```@repl expint
ExpEK(order=3, L=-1) == EK0(prior=IOUP(3, -1))
```
Similarly, the [`RosenbrockExpEK`](@ref) solver is also just a short-hand:
```@repl expint
RosenbrockExpEK(order=3) == EK1(prior=IOUP(3, update_rate_parameter=true))
```

This means that you can also construct other probabilistic exponential integrators by hand!
In this example the `EK1` with `IOUP` prior with rate parameter `-1` performs extremely well:
```@example expint
sol_expek1 = solve(prob, EK1(prior=IOUP(3, -1), diffusionmodel=DM), adaptive=false, dt=STEPSIZE)

plot(ylims=(0.3, 1.05))
plot!(ref, color=:black, linestyle=:dash, label="Reference")
plot!(sol_expek1, denseplot=false, marker=:o, markersize=2, label="EK1 + IOUP")
```


### References

```@bibliography
Pages = []
Canonical = false

bosch23expint
```
