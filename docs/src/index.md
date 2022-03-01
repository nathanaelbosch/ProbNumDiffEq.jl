# ProbNumDiffEq.jl: Probabilistic Numerical Solvers for Differential Equations


![Banner](https://raw.githubusercontent.com/nathanaelbosch/ProbNumDiffEq.jl/main/examples/banner.svg)


__ProbNumDiffEq.jl__ provides _probabilistic numerical_ ODE solvers to the
[DifferentialEquations.jl](https://docs.sciml.ai/stable/) ecosystem.
The implemented _ODE filters_ solve differential equations via Bayesian filtering and smoothing and compute not just a single point estimate of the true solution, but a posterior distribution that contains an estimate of its numerical approximation error.


---

__For more probabilistic numerics check out the [ProbNum](https://www.probabilistic-numerics.org/en/latest/) Python package.__
It implements probabilistic ODE solvers, but also probabilistic linear solvers, Bayesian quadrature, and many filtering and smoothing implementations.

---



## Installation
Run Julia, enter `]` to bring up Julia's package manager, and add the ProbNumDiffEq.jl package:
```
julia> ]
(v1.7) pkg> add ProbNumDiffEq.jl
```

## [Getting Started](@ref)
To quickly try out ProbNumDiffEq.jl check out the "[Getting Started](@ref)" tutorial.


## Features
- Two extended Kalman filtering-based probabilistic solvers: the explicit [`EK0`](@ref) and semi-implicit [`EK1`](@ref).
- Adaptive step-size selection (PI control)
- On-line uncertainty calibration, for multiple different measurement models
- Dense output
- Sampling from the solution
- Callback support
- Convenient plotting through a Plots.jl recipe
- Automatic differentiation via ForwardDiff.jl
- Supports arbitrary precision numbers via BigFloats.jl
- Specialized solvers for second-order ODEs

## Benchmarks
- [Multi-Language Wrapper Benchmark](https://github.com/nathanaelbosch/ProbNumDiffEq.jl/blob/benchmarks/benchmarks/multi-language-wrappers.ipynb):
  ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.

## References
- N. Bosch, P. Hennig, F. Tronarp: **Calibrated Adaptive Probabilistic ODE Solvers** (2021)
- F. Tronarp, S. Särkkä, and P. Hennig: **Bayesian ODE Solvers: The Maximum A Posteriori Estimate** (2021)
- N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020)
- H. Kersting, T. J. Sullivan, and P. Hennig: **Convergence Rates of Gaussian Ode Filters** (2020)
- F. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: **Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective** (2019)
- C. J. Oates and T. J. Sullivan: **A modern retrospective on probabilistic numerics** (2019)
- M. Schober, S. Särkkä, and P. Hennig: **A Probabilistic Model for the Numerical Solution of Initial Value Problems** (2018)
- P. Hennig, M. A. Osborne, and M. Girolami: **Probabilistic numerics and uncertainty in computations** (2015)
A more detailed list of references can be found on the [probabilistic-numerics.org homepage](http://probabilistic-numerics.org/en/latest/research.html).
