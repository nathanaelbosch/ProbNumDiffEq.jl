# Solvers

ProbNumDiffEq.jl provides mainly the following two solvers, both based on extended Kalman filtering and smoothing.
For the best results we suggest using `EK1`, but note that it relies on the Jacobian of the vector field.

!!! note
    All solvers are compatible with DAEs in mass-matrix ODE form, and specialize on second-order ODEs.

```@docs
EK1
EK0
```
