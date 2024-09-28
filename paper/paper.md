---
title: 'ProbNumDiffEq.jl: Probabilistic Numerical Solvers for Ordinary Differential Equations in Julia'
tags:
  - Julia
  - probabilistic numerics
  - differential equations
  - Bayesian filtering and smoothing
  - simulation
authors:
  - name: Nathanael Bosch
    orcid: 0000-0003-0139-4622
    affiliation: 1
affiliations:
 - name: Tübingen AI Center, University of Tübingen, Germany
   index: 1
date: 17 July 2024
bibliography: paper.bib
---

# Summary

Probabilistic numerical solvers have emerged as an efficient framework for simulation, uncertainty quantification, and inference in dynamical systems.
In comparison to traditional numerical methods, which approximate the true trajectory of a system only by a single point estimate, probabilistic numerical solvers compute a _distribution_ over the true unknown solution of the given differential equation
and thereby provide information about the numerical error incurred during the computation.
ProbNumDiffEq.jl implements such probabilistic numerical solvers for ordinary differential equations (ODEs) and differential-algebraic equations (DAEs) in the Julia programming language [@julia] within the DifferentialEquations.jl ecosystem [@rackauckas2017differentialequations].

More concretely, ProbNumDiffEq.jl provides a range of probabilistic numerical solvers for ordinary differential equations based on Bayesian filtering and smoothing,
which have emerged as a particularly efficient and flexible class of methods for solving ODEs [@schober19; @kersting20; @tronarp19].
These so-called "ODE filters" have known polynomial convergence rates 
[@kersting20; @tronarp21]
and numerical stability properties (such as A-stability or L-stability)
[@tronarp19; @bosch2023probabilistic],
their computational complexity is comparable to traditional numerical methods
[@krämer2021highdim], 
they are applicable to a range of numerical differential equation problems 
[@kraemer202bvp; @kraemer22mol; @bosch22pick],
and they can be formulated parallel-in-time 
[@bosch2023parallelintime].
ODE filters also provide a natural framework for ODE parameter inference 
[@kersting20invprob; @tronarp2022fenrir; @schmidt21; @beck2024diffusion].
ProbNumDiffEq.jl implements many of the methods referenced above and packages them in a software library with the aim to be easy-to-use, feature-rich, well-documented, and efficiently implemented.

# Statement of need

Filtering-based probabilistic numerical ODE solvers have been an active field of research for the past decade, but their application in practical simulation and inference problems has been limited.
ProbNumDiffEq.jl aims to bridge this gap.
ProbNumDiffEq.jl implements probabilistic numerical methods as performant, documented, and easy-to-use ODE solvers inside the well-established DifferentialEquations.jl ecosystem [@rackauckas2017differentialequations].
Thereby, the package benefits from the extensive testing, documentation, performance optimization, and functionality that DifferentialEquations.jl provides.
Users can easily find help and examples regarding many features that are not particular to ProbNumDiffEq.jl in the DifferentialEquations.jl documentation, 
and we provide additional examples and tutorials specific to the probabilistic solvers in the ProbNumDiffEq.jl documentation.
We believe that this deep integration within DifferentialEquations.jl is a key feature to attract users to probabilistic numerics by enabling the use of probabilistic ODE solvers as drop-in replacements for traditional ODE solvers.

On the other hand, ProbNumDiffEq.jl also aims to accelerate the development of new probabilistic numerical ODE solvers by providing a solid foundation to both build on and compare against.
Several publications have been developed with ProbNumDiffEq.jl, including contributions on
step-size adaptation and calibration of these solvers [@bosch21capos],
energy-preserving solvers and DAE solvers [@bosch22pick],
probabilistic exponential integrators [@bosch2023probabilistic],
and novel parameter inference algorithms [@tronarp2022fenrir; @beck2024diffusion].
We also hope that by providing documented and performant implementations of published algorithms, we facilitate researchers to use these methods as baselines when developing new numerical solvers.

ProbNumDiffEq.jl is also the only software package in Julia, at the time of writing, that provides a comprehensive set of probabilistic numerical ODE solvers.
Outside of Julia, two other software packages provide a similar functionality.
ProbNum [@wenger2021probnum]
is a Python package that implements probabilistic numerical methods for various numerical problems, including linear systems, quadrature, and ODEs.
ProbNum particularly aims to facilitate rapid experimentation and accelerate the development of new methods [@wenger2021probnum].
It is therefore broader in scope and provides functionality not covered by ProbNumDiffEq.jl, but it also lacks some of the specialized ODE solvers available in ProbNumDiffEq.jl.
In addition, with its reliance on Python and NumPy [@numpy] and the lack of just-in-time compilation, it is also generally less performant.
ProbDiffEq [@probdiffeq]
is a probabilistic numerical ODE solver package built on JAX.
At the time of writing, it provides a very similar set of ODE solvers as ProbNumDiffEq.jl with the addition of certain filtering and smoothing methods and the lack of certain specialized ODE solvers---but as both ProbDiffEq and ProbNumDiffEq.jl are under active development, this might change in the future.
By building on JAX and leveraging its just-in-time compilation capabilities, ProbDiffEq provides ODE solvers with similar performance as those implemented in ProbNumDiffEq.jl (shown through benchmarks in both packages comparing to SciPy [@scipy]).
In summary, ProbNumDiffEq.jl provides one of the most feature-rich and performant probabilistic numerical ODE solver packages currently available and is the only one built on the Julia programming language.

# Acknowledgements

The author gratefully acknowledges co-funding by the European Union (ERC, ANUBIS, 101123955. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council. Neither the European Union nor the granting authority can be held responsible for them). 
The author thanks the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for their support.

I am grateful to Philipp Hennig for his support throughout the development of this package.
I thank Nicholas Krämer, Filip Tronarp, and Jonathan Schmidt for many valuable discussions about probabilistic numerical ODE solvers and their correct, efficient, and elegant implementation.
I also thank Christopher Rackauckas for support and feedback on how to integrate ProbNumDiffEq.jl with the DifferentialEquations.jl ecosystem and for including ProbNumDiffEq.jl into the testing pipeline of OrdinaryDiffEq.jl.
I acknowledge contributions from 
Pietro Monticone (\@pitmonticone),
Vedant Puri (\@vpuri3),
Tim Holy (\@timholy),
Daniel González Arribas (\@DaniGlez),
David Widmann (\@devmotion),
Christopher Rackauckas (\@ChrisRackauckas),
Qingyu Qu (\@ErikQQY),
Cornelius Roemer (\@corneliusroemer),
and Jose Storopoli (\@storopoli).


# References
