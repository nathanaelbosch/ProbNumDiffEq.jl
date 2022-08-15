# Probabilistic Numerical Differential Equation Solvers

![Banner](https://raw.githubusercontent.com/nathanaelbosch/ProbNumDiffEq.jl/main/examples/banner.svg)

__ProbNumDiffEq.jl__ provides _probabilistic numerical_ solvers to the
[DifferentialEquations.jl](https://diffeq.sciml.ai/stable/) ecosystem.
The implemented _ODE filters_ solve differential equations via Bayesian filtering and smoothing and compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.

For a short intro video, check out our [poster presentation at JuliaCon2021](https://www.youtube.com/watch?v=EMFl6ytP3iQ).

* * *

__For more probabilistic numerics check out the [ProbNum](https://probnum.readthedocs.io/en/latest/) Python package.__
It implements probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.

* * *

## Installation

Run Julia, enter `]` to bring up Julia's package manager, and add the ProbNumDiffEq.jl package:

```
julia> ]
(v1.7) pkg> add ProbNumDiffEq.jl
```

## Getting Started

For a quick introduction check out the "[Solving ODEs with Probabilistic Numerics](@ref)" tutorial.

## Features

  - Two extended Kalman filtering-based probabilistic solvers: the explicit [`EK0`](@ref) and semi-implicit [`EK1`](@ref).
  - Adaptive step-size selection (by default with PI control)
  - On-line uncertainty calibration, for multiple different measurement models
  - Dense output
  - Sampling from the solution
  - Callback support
  - Convenient plotting through a Plots.jl recipe
  - Automatic differentiation via ForwardDiff.jl
  - Supports arbitrary precision numbers via BigFloats.jl
  - Specialized solvers for second-order ODEs (demo will be added)
  - Compatible with DAEs in mass-matrix ODE form (demo will be added)

## Benchmarks

  - [Multi-Language Wrapper Benchmark](https://nbviewer.org/github/nathanaelbosch/ProbNumDiffEq.jl/blob/main/benchmarks/multi-language-wrappers.ipynb):
    ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.

## [References](@id references)

  - N. Bosch, F. Tronarp, P. Hennig: **Pick-and-Mix Information Operators for Probabilistic ODE Solvers** (2022)
  - N. Krämer, N. Bosch, J. Schmidt, P. Hennig: **Probabilistic ODE Solutions in Millions of Dimensions** (2021)
  - N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
  - F. Tronarp, S. Särkkä, and P. Hennig: **Bayesian ODE Solvers: The Maximum A Posteriori Estimate** (2021)
  - N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020)
  - H. Kersting, T. J. Sullivan, and P. Hennig: **Convergence Rates of Gaussian Ode Filters** (2020)
  - F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019)
  - M. Schober, S. Särkkä, and P. Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems** (2018)

A much more detailed list of references, not only on ODE filters but on probabilistic numerics in general, can be found on the [probabilistic-numerics.org homepage](https://www.probabilistic-numerics.org/research/general/).
