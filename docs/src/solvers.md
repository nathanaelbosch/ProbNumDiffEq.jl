# Solvers and Options

ProbNumDiffEq.jl provides mainly the following two solvers, both based on extended Kalman filtering and smoothing.
For the best results we suggest using `EK1`, but note that it requires that the Jacobian of the vector field is defined.

```@docs
EK1
EK0
```

#### Experimental: Iterated extended Kalman smoothing
We do not recommend using the following solver, but if you are interested feel free to open an issue!
```@docs
IEKS
solve_ieks
```
