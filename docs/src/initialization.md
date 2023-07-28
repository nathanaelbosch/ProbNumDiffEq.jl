# Initialization

The notion of "initialization" relates to the _prior_ part of the model.


## Background: The prior

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
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= \textcolor{#389826}{A} Y(t) \ \text{d}t + \textcolor{#4063D8}{\Gamma} \ \text{d}W(t), \\
Y(0) &\sim \textcolor{#9558B2}{ \mathcal{N} \left( \mu_0, \Sigma_0 \right) }.
\end{aligned}
```
Then ``Y^{(i)}(t)`` models the ``i``-th derivative of ``y(t)``.
**In this section, we consider the _initial distribution_ ``\textcolor{purple}{ \mathcal{N} \left( \mu_0, \Sigma_0 \right) }``.**
If you're more interested in the
_drift matrix_ ``\textcolor{#389826}{A}``
check out the [Priors](@ref) section,
and for more info on
the _diffusion_ ``\textcolor{#4063D8}{\Gamma}``
check out the
[Diffusion models and calibration](@ref) section.


## Setting the initial distribution

Let's assume an initial value problem of the form
```math
\begin{aligned}
\dot{y}(t) &= f(y(t), t), \qquad [0, T], \\
y(0) &= y_0.
\end{aligned}
```
It is clear that this contains quite some information for ``Y(0)``:
The initial value ``y_0`` and the vector field ``f`` imply
```math
\begin{aligned}
Y^{(0)}(0) &= y_0, \\
Y^{(1)}(0) &= f(y_0, 0).
\end{aligned}
```
It turns out that we can also compute higher-order derivatives of ``y`` with the chain rule,
and then use these to better initialize ``Y^{(i)}(0)``.
This, done efficiently with Taylor-mode autodiff by using
[TaylorIntegration.jl](https://perezhz.github.io/TaylorIntegration.jl/latest/),
is what [`TaylorModeInit`](@ref) does.
See also [[1]](@ref initrefs).

**In the vast majority of cases, just stick to the exact Taylor-mode initialization [`TaylorModeInit`](@ref)!**


## API

```@docs
TaylorModeInit
ClassicSolverInit
```


## References

```@bibliography
Pages = []
Canonical = false

kraemer20stableimplementation
schober16probivp
```
