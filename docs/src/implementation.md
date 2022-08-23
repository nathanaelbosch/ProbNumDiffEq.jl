# Solver Implementation via OrdinaryDiffEq.jl

ProbNumDiffEq.jl builds directly on OrdinaryDiffEq.jl to benefit from its iterator interface, flexible step-size control, and efficient Jacobian calculations.
But, this requires extending non-public APIs.
This page is meant to provide an overview on which parts exactly ProbNumDiffEq.jl builds on.

For more discussion on the pros and cons of building on OrdinaryDiffEq.jl, see
[this thread on discourse](https://discourse.julialang.org/t/building-on-ordinarydiffeq-jl-vs-diffeqbase-jl/85620/4).

## Building on OrdinaryDiffEq.jl

ProbNumDiffEq.jl shares *most* of OrdinaryDiffEq.jl's implementation.
In particular:
1. `OrdinaryDiffEq.__init` builds the cache and the integrator, and calls `OrdinaryDiffEq.initialize!`
2. `OrdinaryDiffEq.solve!` implements the actual iterator structure, with
   - `OrdinaryDiffEq.loopheader!`
   - `OrdinaryDiffEq.perform_step!`
   - `OrdinaryDiffEq.loopfooter!`
   - `OrdinaryDiffEq.postamble!`

ProbNumDiffEq.jl builds around this structure and overloads some of the parts:

- **Algorithms:** `EK0/EK1 <: AbstractEK <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm`
  - `./src/algorithms.jl` provides the algorithms themselves
  - `./src/alg_utils.jl` implements many traits (e.g. relating to autodiff, implicitness, step-size control)
- **Cache:** `EKCache <: AbstractODEFilterCache <: OrdinaryDiffEq.OrdinaryDiffEqCache`
  - `./src/caches.jl` implements the cache and its main constructor: `OrdinaryDiffEq.alg_cache`
- **Initialization and `perform_step!`:** via `OrdinaryDiffEq.initialize!` and `OrdinaryDiffEq.perform_step!`.   
  Implemented in `./src/perform_step.jl`.
- **Custom postamble** by overloading `OrdinaryDiffEq.postamble!` (which should always call `OrdinaryDiffEq._postamble!`).
  This is where we do the "smoothing" of the solution. 
  Implemented in `./src/integrator_utils.jl`. 
- **Custom saving** by overloading `OrdinaryDiffEq.savevalues!` (which should always call `OrdinaryDiffEq._savevalues!`). 
  Implemented in `./src/integrator_utils.jl`.


## Building on DiffEqBase.jl

- **`DiffEqBase.__init`** is currently overloaded to transform OOP problems into IIP problems (in `./src/solve.jl`).
- **The solution object:** `ProbODESolution <: AbstractProbODESolution <: DiffEqBase.AbstractODESolution`
  - `./src/solution.jl` implements the main parts.
    Note that the main constructor `DiffEqBase.build_solution` is called by `OrdinaryDiffEq.__init` - so OrdinaryDiffEq.jl has control over its inputs.
  - There is also `MeanProbODESolution <: DiffEqBase.AbstractODESolution`: It allows handling the mean of a probabilistic ODE solution the same way one would handle any "standard" ODE solution - e.g. it is compatible with `DiffEqDevTools.appxtrue`.
  - `AbstractODEFilterPosterior <: DiffEqBase.AbstractDiffEqInterpolation` is the current interpolant, but it does not actually fully handle the interpolation right now. This part might be subject to change soon.
  - *Plot recipe* in `./src/solution_plotting.jl`
  - *Sampling* in `./src/solution_sampling.jl**

## Other packages
- `DiffEqDevTools.appxtrue` is overloaded to work with `ProbODESolution` (by just doing `mean(sol)`). This also enables `DiffEqDevTools.WorkPrecision` to work out of th box.
