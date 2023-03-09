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
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= \textcolor{forestgreen}{A} Y(t) \ \text{d}t + \textcolor{royalblue}{\Gamma} \ \text{d}W(t).
\end{aligned}
```
Then ``Y^{(i)}(t)`` models the ``i``-th derivative of ``y(t)``.
**In this section, we consider choices relating to the _"diffusion"_ ``\textcolor{royalblue}{\Gamma}``.**
If you're more interested in the _drift matrix_``\textcolor{forestgreen}{A}``, check out the [Priors](@ref) section.


## Diffusion and calibration

We call ``\textcolor{royalblue}{\Gamma}`` the _"diffusion"_ parameter.
Since it is typically not known we need to estimate it; this is called _"calibration"_.

There are a few different choices for how to model and estimate ``\textcolor{royalblue}{\Gamma}``:
- [`FixedDiffusion`](@ref) assumes an isotropic, time-fixed ``\textcolor{royalblue}{\Gamma} = \sigma \cdot I_d``,
- [`DynamicDiffusion`](@ref) assumes an isotropic, time-varying ``\textcolor{royalblue}{\Gamma}(t) = \sigma(t) \cdot I_d``,
- [`FixedMVDiffusion`](@ref) assumes a diagonal, time-fixed ``\textcolor{royalblue}{\Gamma} = \operatorname{diag}(\sigma_1, \dots, \sigma_d)``,
- [`DynamicMVDiffusion`](@ref) assumes a diagonal, time-varying ``\textcolor{royalblue}{\Gamma}(t) = \operatorname{diag}(\sigma_1(t), \dots, \sigma_d(t))``.

Or more compactly:

|              | Isotropic:                   | Diagonal (only for the `EK0`(@ref)) |
|--------------|----------------------------|-------------------------------------|
| Time-varying | [`DynamicDiffusion`](@ref) | [`DynamicMVDiffusion`](@ref)        |
| Time-fixed   | [`FixedDiffusion`](@ref)   | [`FixedMVDiffusion`](@ref)          |


For more details on diffusions and calibration, check out this paper [[1]](@ref diffusionrefs).


## API

```@docs
DynamicDiffusion
FixedDiffusion
DynamicMVDiffusion
FixedMVDiffusion
```


## [References](@id diffusionrefs)

[1] N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021) ([link](http://proceedings.mlr.press/v130/bosch21a.html))
