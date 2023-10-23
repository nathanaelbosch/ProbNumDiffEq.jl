# Priors

**TL;DR: If you're unsure which prior to use, just stick to the default integrated Wiener process prior [`IWP`](@ref)!**

# Background

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

## API

```@docs
IWP
IOUP
Matern
```
