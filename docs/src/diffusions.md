# Diffusion models and calibration

The notion of "diffusion" and "calibration" relates to the _prior_ part of the model.

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
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1, \\
\text{d} Y^{(q)}(t) &= \textcolor{#389826}{A} Y(t) \ \text{d}t + \textcolor{#4063D8}{\Gamma} \ \text{d}W(t), \\
Y(0) &\sim \textcolor{purple}{ \mathcal{N} \left( \mu_0, \Sigma_0 \right) }.
\end{aligned}
```
Then ``Y^{(i)}(t)`` models the ``i``-th derivative of ``y(t)``.
**In this section, we consider choices relating to the _"diffusion"_ ``\textcolor{#4063D8}{\Gamma}``.**
If you're more interested in the _drift matrix_ ``\textcolor{#389826}{A}`` check out the [Priors](@ref) section,
and for info on the initial distribution ``\textcolor{purple}{ \mathcal{N} \left( \mu_0, \Sigma_0 \right) }`` check out the [Initialization](@ref) section.


## Diffusion and calibration

We call ``\textcolor{#4063D8}{\Gamma}`` the _"diffusion"_ parameter.
Since it is typically not known we need to estimate it; this is called _"calibration"_.

There are a few different choices for how to model and estimate ``\textcolor{#4063D8}{\Gamma}``:
- [`FixedDiffusion`](@ref) assumes an isotropic, time-fixed ``\textcolor{#4063D8}{\Gamma} = \sigma \cdot I_d``,
- [`DynamicDiffusion`](@ref) assumes an isotropic, time-varying ``\textcolor{#4063D8}{\Gamma}(t) = \sigma(t) \cdot I_d`` (**recommended**),
- [`FixedMVDiffusion`](@ref) assumes a diagonal, time-fixed ``\textcolor{#4063D8}{\Gamma} = \operatorname{diag}(\sigma_1, \dots, \sigma_d)``,
- [`DynamicMVDiffusion`](@ref) assumes a diagonal, time-varying ``\textcolor{#4063D8}{\Gamma}(t) = \operatorname{diag}(\sigma_1(t), \dots, \sigma_d(t))``.

Or more compactly:

|              | Isotropic:                 | Diagonal (only for the [`EK0`](@ref)) |
|--------------|----------------------------|---------------------------------------|
| Time-varying | [`DynamicDiffusion`](@ref) | [`DynamicMVDiffusion`](@ref)          |
| Time-fixed   | [`FixedDiffusion`](@ref)   | [`FixedMVDiffusion`](@ref)            |


For more details on diffusions and calibration, check out this paper [bosch20capos](@cite).


## API

```@docs
DynamicDiffusion
FixedDiffusion
DynamicMVDiffusion
FixedMVDiffusion
```


## [References](@id diffusionrefs)

```@bibliography
Pages = []
Canonical = false

bosch20capos
```
