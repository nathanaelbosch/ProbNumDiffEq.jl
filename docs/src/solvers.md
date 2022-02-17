# Solvers and Options

ProbNumDiffEq.jl provides mainly the following two solvers, both based on extended Kalman filtering and smoothing.
For the best results we suggest using `EK1`, but note that it relies on the Jacobian of the vector field.

```@docs
EK1
EK0
```
