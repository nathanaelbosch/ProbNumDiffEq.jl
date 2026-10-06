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
   - `SciMLBase.postamble!`

ProbNumDiffEq.jl builds around this structure and overloads some of the parts:

- **Algorithm:** `ODEFilter <: OrdinaryDiffEq.OrdinaryDiffEqAdaptiveAlgorithm`
  - `./src/algorithms.jl` provides the algorithm; `EK0`, `EK1` and `DiagonalEK1` are functions that return an `ODEFilter` and differ only in its `linearization`
  - `./src/alg_utils.jl` implements many traits (relating to automatic differentiation, step-size control, etc), from the fields of the `ODEFilter`
- **Cache:** `EKCache <: AbstractODEFilterCache <: OrdinaryDiffEq.OrdinaryDiffEqCache`
  - `./src/caches.jl` implements the cache and its main constructor: `OrdinaryDiffEq.alg_cache`
- **Initialization and `perform_step!`:** via `OrdinaryDiffEq.initialize!` and `OrdinaryDiffEq.perform_step!`.
  Implemented in `./src/perform_step.jl`.
- **Custom postamble** by overloading `SciMLBase.postamble!` (which should always call `OrdinaryDiffEqCore._postamble!`).
  This is where we do the "smoothing" of the solution.
  Implemented in `./src/integrator_utils.jl`.
- **Custom saving** by overloading `OrdinaryDiffEq.savevalues!` (which should always call `OrdinaryDiffEq._savevalues!`).
  Implemented in `./src/integrator_utils.jl`.


## Building on DiffEqBase.jl

- **`DiffEqBase.__init`** is currently overloaded to transform OOP problems into IIP problems (in `./src/solve.jl`).
- **The solution object:** `ProbODESolution <: AbstractProbODESolution <: SciMLBase.AbstractODESolution`
  - `./src/solution.jl` implements the main parts.
    Note that the main constructor `SciMLBase.build_solution` is called by `OrdinaryDiffEq.__init`, so OrdinaryDiffEq.jl has control over its inputs.
  - `MeanProbODESolution <: SciMLBase.AbstractODESolution` is a wrapper that allows handling the mean of a probabilistic ODE solution the same way one would handle any "standard" ODE solution, by just ignoring the covariances.
  - `AbstractODEFilterPosterior <: SciMLBase.AbstractDiffEqInterpolation` handles the interpolation.
  - *Plot recipe* in `./ext/RecipesBaseExt.jl`
  - *Sampling* in `./src/solution_sampling.jl`
- `DiffEqBase.prepare_alg(::ODEFilter)`; closely follows a similar function implemented in OrdinaryDiffEq.jl `./src/alg_utils.jl`
   - it relies on SciMLBase's generic `remake`, through the keyword constructor of the `ODEFilter`

## Other packages
- `DiffEqDevTools.appxtrue`: We extend this function to work with `ProbODESolution`. This also enables `DiffEqDevTools.WorkPrecision` to work out of the box.
