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

**If you're unsure which prior to use, just stick to the integrated Wiener process prior [`IWP`](@ref)!**
This is also the default choice for all solvers.
The other priors are rather experimental / niche at the time of writing.

## Prior choices

ProbNumDiffEq.jl currently supports
integrated Wiener processes ([`IWP`](@ref)),
integrated Ornstein--Uhlenbeck processes ([`IOUP`](@ref)), and
Matérn processes ([`Matern`](@ref)) as priors for the solvers.
Let's look at each of them in turn and visualize some examples.

```@example priors
using ProbNumDiffEq, Plots
plotrange = range(0, 10, length=250)
```

### Integrated Wiener process prior [`IWP`](@ref)
This is the default prior for all solvers, and it is the most common choice in the literature.

Two-times integrated Wiener process to model a one-dimensional ODE solution:
```@example priors
plot(IWP(1), plotrange; ylims=(-10,10))
```

Four-times integrated Wiener process to model a two-dimensional ODE solution:
```@example priors
plot(IWP(4), plotrange; ylims=(-10,10))
```

### Integrated Ornstein--Uhlenbeck process prior [`IOUP`](@ref)
This prior is mostly used in the context of [Probabilistic Exponential Integrators](@ref probexpinttutorial).


```@example priors
plot(IOUP(1, -1), plotrange; ylims=(-10, 10))
```

```@example priors
plot(IOUP(1, 1), plotrange; ylims=(-10, 10))
```

```@example priors
rate_parameter = [-0.2 -2π; 2π -0.2]
plot(IOUP(2, 1, rate_parameter), plotrange)

```

### Matérn process prior [`Matern`](@ref)
Matérn processes are well-known in the Gaussian process literature, but not much explored for probabilistic ODE solvers.

```@example priors
plot(Matern(2, 1), plotrange)
```
```@example priors
plot(Matern(2, 10), plotrange)
```
```@example priors
plot(Matern(2, 0.1), plotrange)
```

## API
```@docs
IWP
IOUP
Matern
```

### Internals
```@docs
ProbNumDiffEq.AbstractGaussMarkovProcess
ProbNumDiffEq.LTISDE
ProbNumDiffEq.wiener_process_dimension
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
