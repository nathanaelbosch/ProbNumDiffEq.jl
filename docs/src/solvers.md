# Solvers and Options

ProbNumDiffEq.jl provides mainly the following two solvers, both based on extended Kalman filtering and smoothing.
For the best results we suggest using `EK1`, but note that it relies on the Jacobian of the vector field.

## Solvers

All solvers are compatible with DAEs in mass-matrix ODE form, and specialize on second-order ODEs.

```@docs
EK1
EK0
```

## Diffusion models and calibration

In a nutshell:
"Dynamic" diffusion models allow the diffusion to change in-between each solver step and are recommended in combination with adaptive step-sizes.
"Fixed" diffusion models keep the diffusion constant and can be helpful in fixed-step settings or for debugging by reducing the complexity of the models.

For more information on the influence of diffusions, check out

  - N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)

```@docs
DynamicDiffusion
FixedDiffusion
DynamicMVDiffusion
FixedMVDiffusion
```

## Initialization

```@docs
TaylorModeInit
ClassicSolverInit
```

## Priors

```@docs
IWP
IOUP
```
