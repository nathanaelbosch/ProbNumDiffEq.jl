# Solvers

ProbNumDiffEq.jl provides two solvers: the [`EK1`](@ref) and the [`EK0`](@ref). Both based on extended Kalman filtering and smoothing, but the latter relies on evaluating the Jacobian of the vector field.
**For the best results, use the [`EK1`](@ref).**

All solvers are compatible with DAEs in mass-matrix ODE form.
They also specialize on second-order ODEs: If the problem is of type [`SecondOrderODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/#SciMLBase.SecondOrderODEProblem), it solves the second-order problem directly; this is more efficient than solving the transformed first-order problem and provides more meaningful posteriors
[[1]](@ref solversrefs).


## API

```@docs
EK1
EK0
ExpEK
RosenbrockExpEK
```

## [References](@id solversrefs)

[1] N. Bosch, F. Tronarp, P. Hennig: **Pick-and-Mix Information Operators for Probabilistic ODE Solvers** (2022) ([link](https://proceedings.mlr.press/v151/bosch22a.html))
