# Probabilistic Numerical Differential Equation Solvers

![Banner](https://raw.githubusercontent.com/nathanaelbosch/ProbNumDiffEq.jl/main/examples/banner.svg)

__ProbNumDiffEq.jl__ provides _probabilistic numerical_ solvers to the
[DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) ecosystem.
The implemented _ODE filters_ solve differential equations via Bayesian filtering and smoothing and compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.

For a short intro video, check out our [poster presentation at JuliaCon2021](https://www.youtube.com/watch?v=EMFl6ytP3iQ).

## Installation

Run Julia, enter `]` to bring up Julia's package manager, and add the ProbNumDiffEq.jl package:

```
julia> ]
(v1.10) pkg> add ProbNumDiffEq
```

## Getting Started

For a quick introduction check out the "[Solving ODEs with Probabilistic Numerics](@ref)" tutorial.

## Features

- Two extended Kalman filtering-based probabilistic solvers: the explicit [`EK0`](@ref) and semi-implicit [`EK1`](@ref).
- Adaptive step-size selection with PI control;
  fully compatible with [DifferentialEquations.jl's timestepping options](https://docs.sciml.ai/DiffEqDocs/stable/extras/timestepping/)
- Online uncertainty calibration for multiple different diffusion models (see "[Diffusion models and calibration](@ref)")
- [Dense output](@ref)
- Sampling from the solution
- Callback support
- Convenient plotting through a [Plots.jl](https://docs.juliaplots.org/latest/) recipe
- Automatic differentiation via [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)
- Arbitrary precision via Julia's built-in [arbitrary precision arithmetic](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic)
- Specialized solvers for second-order ODEs (see [Second Order ODEs and Energy Preservation](@ref))
- Compatible with DAEs in mass-matrix ODE form (see [Solving DAEs with Probabilistic Numerics](@ref))
- Data likelihoods for parameter-inference in ODEs (see [Parameter Inference with ProbNumDiffEq.jl](@ref))


## Related packages

- [probdiffeq](https://pnkraemer.github.io/probdiffeq/): Fast and feature-rich filtering-based probabilistic ODE solvers in JAX.
- [ProbNum](https://probnum.readthedocs.io/en/latest/): Probabilistic numerics in Python. It has not only probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.
