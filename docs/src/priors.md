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

ProbNumDiffEq.jl currently supports three classes of priors for the solvers:
- ``q``-times integrated Wiener processes ([`IWP`](@ref))
- ``q``-times integrated Ornstein--Uhlenbeck processes ([`IOUP`](@ref))
- Matérn processes ([`Matern`](@ref))
Let's look at each of them in turn and visualize some examples.

```@example priors
using ProbNumDiffEq, Plots
plotrange = range(0, 10, length=250)
nothing # hide
```

### Integrated Wiener process ([`IWP`](@ref))
The ``q``-times integrated Wiener process prior [`IWP`](@ref) is the most common prior choice in the probabilistic ODE solver literature,
and is the default choice for the solvers in ProbNumDiffEq.jl.

Here is how the [`IWP`](@ref) looks for varying smoothness parameters ``q``:
```@example priors
plot(
    plot(IWP(1), plotrange; title="q=1"),
    plot(IWP(2), plotrange; title="q=2"),
    plot(IWP(3), plotrange; title="q=3"),
    plot(IWP(4), plotrange; title="q=4");
    ylims=(-20,20),
)
```


### Integrated Ornstein--Uhlenbeck process ([`IOUP`](@ref))
The ``q``-times integrated Ornstein--Uhlenbeck process prior [`IOUP`](@ref) is a generalization of the IWP prior,
where the drift matrix ``\textcolor{#389826}{A}`` is not zero, but is of the form
```math
\begin{aligned}
\textcolor{#389826}{A} = \begin{bmatrix} 0 & 0 & 0 & \dots & R \end{bmatrix},
\end{aligned}
```
where ``R`` is a ``q \times q`` matrix called the "rate parameter" of the prior.
That is, the ``q``-th derivative of the solution is not just driven by the Wiener process, but also by a linear ODE with drift ``R``.
This prior is mostly used in the context of [Probabilistic Exponential Integrators](@ref probexpinttutorial) to include the linear part of a semi-linear ODE in the prior.

Here is how the [`IOUP`](@ref) looks for varying rate parameters:
```@example priors
plot(
    plot(IOUP(1, -1), plotrange; title="q=1,R=-1", ylims=(-20,20)),
    plot(IOUP(1, 1), plotrange; title="q=1,R=1", ylims=(-20,20)),
    plot(IOUP(4, -1), plotrange; title="q=4,R=-1", ylims=(-50,50)),
    plot(IOUP(4, 1), plotrange; title="q=4,R=1", ylims=(-50,50));
)
```

In the context of [Probabilistic Exponential Integrators](@ref probexpinttutorial), the rate parameter is often chosen according to the given ODE.
Here is an example for a damped oscillator:
```@example priors
plot(IOUP(2, 1, [-0.2 -2π; 2π -0.2]), plotrange; plot_title="damped oscillator prior");
```

### Matérn process ([`Matern`](@ref))
The class of [`Matern`](@ref) processes is well-known in the Gaussian process literature.
These processes also have a corresponding SDE representation as explained in the background section above,
with a specific drift matrix ``\textcolor{#389826}{A}`` [sarkka19appliedsde](@cite).

Here is how the [`Matern`](@ref) looks for varying smoothness parameters ``q``:
```@example priors
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
IWP
IOUP
Matern
```

### Internals
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
