# Priors

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
\text{d} Y^{(q)}(t) &= \textcolor{forestgreen}{A} Y(t) \ \text{d}t + \textcolor{royalblue}{\Gamma} \ \text{d}W(t).
\end{aligned}
```
Then ``Y^{(i)}(t)`` models the ``i``-th derivative of ``y(t)``.
**In this section, we consider choices relating to the _drift matrix_ ``\textcolor{forestgreen}{A}``.**
If you're more interested in the _diffusion_ ``\textcolor{royalblue}{\Gamma}``, check out the [Diffusion models and calibration](@ref) section.

**If you're unsure which prior to use, just stick to the integrated Wiener process prior [`IWP`](@ref)!**
This is also the default choice for all solvers.
The other priors are rather experimental / niche at the time of writing.

## API

```@docs
IWP
IOUP
Matern
```
