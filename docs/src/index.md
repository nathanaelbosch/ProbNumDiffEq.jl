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
(v1.9) pkg> add ProbNumDiffEq
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


## [References](@id references)

The main references _for this package_ include:

- N. Bosch, P. Hennig, F. Tronarp:
  [**Probabilistic Exponential Integrators**](https://arxiv.org/abs/2305.14978)
  (2023)
- N. Bosch, F. Tronarp, P. Hennig:
  [**Pick-and-Mix Information Operators for Probabilistic ODE Solvers**](https://proceedings.mlr.press/v151/bosch22a.html)
  (2022)
- N. Krämer, N. Bosch, J. Schmidt, P. Hennig:
  [**Probabilistic ODE Solutions in Millions of Dimensions**](https://proceedings.mlr.press/v162/kramer22b.html)
  (2022)
- N. Bosch, P. Hennig, F. Tronarp:
  [**Calibrated Adaptive Probabilistic ODE Solvers**](http://proceedings.mlr.press/v130/bosch21a.html)
  (2021)
- F. Tronarp, S. Särkkä, and P. Hennig:
  [**Bayesian ODE Solvers: The Maximum A Posteriori Estimate**](https://link.springer.com/article/10.1007/s11222-021-09993-7)
  (2021)
- N. Krämer, P. Hennig:
  [**Stable Implementation of Probabilistic ODE Solvers**](https://arxiv.org/abs/2012.10106v1)
  (2020)
- H. Kersting, T. J. Sullivan, and P. Hennig:
  [**Convergence Rates of Gaussian Ode Filters**](https://link.springer.com/article/10.1007/s11222-020-09972-4)
  (2020)
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig:
  [**Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective**](https://link.springer.com/article/10.1007/s11222-019-09900-1)
  (2019)
- M. Schober, S. Särkkä, and P. Hennig:
  [**A Probabilistic Model for the Numerical Solution of Initial Value Problems**](https://link.springer.com/article/10.1007/s11222-017-9798-7)
  (2018)

More references on ODE filters and on probabilistic numerics in general can be found on [probabilistic-numerics.org ](https://www.probabilistic-numerics.org/research/general/).


## Related packages

- [probdiffeq](https://pnkraemer.github.io/probdiffeq/): Fast and feature-rich filtering-based probabilistic ODE solvers in JAX.
- [ProbNum](https://probnum.readthedocs.io/en/latest/): Probabilistic numerics in Python. It has not only probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.
- [Fenrir.jl](https://github.com/nathanaelbosch/Fenrir.jl): Parameter-inference in ODEs with probabilistic ODE solvers. This package builds on ProbNumDiffEq.jl to provide a negative marginal log-likelihood function, which can then be used with an optimizer or with MCMC for parameter inference.
