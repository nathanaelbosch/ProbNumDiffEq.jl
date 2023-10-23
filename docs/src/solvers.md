# Solvers

ProbNumDiffEq.jl provides two solvers: the [`EK1`](@ref) and the [`EK0`](@ref). Both based on extended Kalman filtering and smoothing, but the latter relies on evaluating the Jacobian of the vector field.

**Which solver should I use?**
- Use the [`EK1`](@ref) to get the best uncertainty quantification and to solve stiff problems.
- Use the [`EK0`](@ref) to get the fastest runtimes and to solve high-dimensional problems.

All solvers are compatible with DAEs in mass-matrix ODE form.
They also specialize on second-order ODEs: If the problem is of type [`SecondOrderODEProblem`](https://docs.sciml.ai/DiffEqDocs/stable/types/dynamical_types/#SciMLBase.SecondOrderODEProblem), it solves the second-order problem directly; this is more efficient than solving the transformed first-order problem and provides more meaningful posteriors
[[1]](@ref solversrefs).

## API
```@docs
EK1
EK0
```

### Probabilistic Exponential Integrators
```@docs
ExpEK
RosenbrockExpEK
```

## [References](@id solversrefs)


```@bibliography
Pages = []
Canonical = false

tronarp18probsol
krämer21highdim
bosch23expint
```
