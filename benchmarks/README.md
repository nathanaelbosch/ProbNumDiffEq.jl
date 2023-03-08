# Benchmarks


- **Multi-Language Wrapper Benchmark**:
  ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.
  Results either [here in the docs](https://nathanaelbosch.github.io/ProbNumDiffEq.jl/dev/benchmarks/multi-language-wrappers/),
  or [directly on github](../docs/src/benchmarks/multi-language-wrappers.md).


## How to run the benchmarks

You can generate interactive notebooks by running
```julia
using Weave
notebook("multi-language-wrappers.jmd")
```
where you can replace `multi-language-wrappers.jmd` with the benchmark of your choice.

The markdown files from the documentation are generated with `runall.jl`
