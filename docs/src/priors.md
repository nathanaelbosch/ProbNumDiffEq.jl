# Priors

**TL;DR: If you're unsure which prior to use, just stick to the default integrated Wiener process prior [`IWP`](@ref)!**

## Background

We model the ODE solution ``y(t)`` with a Gauss--Markov prior.
More precisely, let
```math
\begin{aligned}
Y(t) = \left[ Y^{(0)}(t), Y^{(1)}(t), \dots Y^{(q)}(t) \right],
\end{aligned}
```
be the solution to the SDE
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1, \\
\text{d} Y^{(q)}(t) &= \textcolor{#389826}{A} Y(t) \ \text{d}t + \textcolor{#4063D8}{\Gamma} \ \text{d}W(t), \\
Y(0) &\sim \textcolor{purple}{ \mathcal{N} \left( \mu_0, \Sigma_0 \right) }.
\end{aligned}
```
Then ``Y^{(i)}(t)`` models the ``i``-th derivative of ``y(t)``.
**In this section, we consider choices relating to the _drift matrix_ ``\textcolor{#389826}{A}``.**
If you're more interested in the _diffusion_ ``\textcolor{#4063D8}{\Gamma}`` check out the [Diffusion models and calibration](@ref) section,
and for info on the initial distribution ``\textcolor{purple}{ \mathcal{N} \left( \mu_0, \Sigma_0 \right) }`` check out the [Initialization](@ref) section.

!!! info
    **If you're unsure which prior to use, just stick to the integrated Wiener process prior [`IWP`](@ref)!**
    This is also the default choice for all solvers.
    The other priors are rather experimental / niche at the time of writing.

## Prior choices

ProbNumDiffEq.jl currently supports three classes of priors for the solvers:
- [`IWP`](@ref): ``q``-times integrated Wiener processes
- [`IOUP`](@ref): ``q``-times integrated Ornstein--Uhlenbeck processes
- [`Matern`](@ref): Matérn processes
Let's look at each of them in turn and visualize some examples.


### Integrated Wiener process ([`IWP`](@ref))
```@docs
IWP
```
Here is how the [`IWP`](@ref) looks for varying smoothness parameters ``q``:
```@example priors
using ProbNumDiffEq, Plots
plotrange = range(0, 10, length=250)
plot(
    plot(IWP(1), plotrange; title="q=1"),
    plot(IWP(2), plotrange; title="q=2"),
    plot(IWP(3), plotrange; title="q=3"),
    plot(IWP(4), plotrange; title="q=4");
    ylims=(-20,20),
)
```
In the context of ODE solvers, the smoothness parameter ``q`` influences the convergence rate of the solver,
and so it is typically chose similarly to the order of a Runge--Kutta method: lower order for low accuracy, higher order for high accuracy.


### Integrated Ornstein--Uhlenbeck process ([`IOUP`](@ref))
The ``q``-times integrated Ornstein--Uhlenbeck process prior [`IOUP`](@ref) is a generalization of the IWP prior, where the drift matrix ``\textcolor{#389826}{A}`` is not zero:
```@docs
IOUP
```

Here is how the [`IOUP`](@ref) looks for varying rate parameters:
```@example priors
using ProbNumDiffEq, Plots
plotrange = range(0, 10, length=250)
plot(
    plot(IOUP(1, -1), plotrange; title="q=1,L=-1", ylims=(-20,20)),
    plot(IOUP(1, 1), plotrange; title="q=1,L=1", ylims=(-20,20)),
    plot(IOUP(4, -1), plotrange; title="q=4,L=-1", ylims=(-50,50)),
    plot(IOUP(4, 1), plotrange; title="q=4,L=1", ylims=(-50,50));
)
```

In the context of [Probabilistic Exponential Integrators](@ref probexpinttutorial), the rate parameter is often chosen according to the given ODE.
Here is an example for a damped oscillator:
```@example priors
plot(IOUP(2, 1, [-0.2 -2π; 2π -0.2]), plotrange; plot_title="damped oscillator prior")
```

### Matérn process ([`Matern`](@ref))
```@docs
Matern
```

Here is how the [`Matern`](@ref) looks for varying smoothness parameters ``q``:
```@example priors
using ProbNumDiffEq, Plots
plotrange = range(0, 10, length=250)
plot(
    plot(Matern(1, 1), plotrange; title="q=1"),
    plot(Matern(2, 1), plotrange; title="q=2"),
    plot(Matern(3, 1), plotrange; title="q=3"),
    plot(Matern(4, 1), plotrange; title="q=4");
)
```
and for varying length scales ``\ell``:
```@example priors
plot(
    plot(Matern(2, 5), plotrange; title="l=5"),
    plot(Matern(2, 2), plotrange; title="l=2"),
    plot(Matern(2, 1), plotrange; title="l=1"),
    plot(Matern(2, 0.5), plotrange; title="l=0.5"),
)
```

## API
```@docs
ProbNumDiffEq.AbstractGaussMarkovProcess
ProbNumDiffEq.LTISDE
ProbNumDiffEq.dim
ProbNumDiffEq.num_derivatives
ProbNumDiffEq.to_sde
ProbNumDiffEq.discretize
ProbNumDiffEq.initial_distribution
```

### Convenience functions to analyze and visualize priors
```@docs
ProbNumDiffEq.marginalize
ProbNumDiffEq.sample
```
