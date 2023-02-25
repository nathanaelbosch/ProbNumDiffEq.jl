# Benchmarks


- **Multi-Language Wrapper Benchmark**:
  ProbNumDiffEq.jl vs. OrdinaryDiffEq.jl, Hairer's FORTRAN solvers, Sundials, LSODA, MATLAB, and SciPy.
  Results in the [jupyter notebook](./multi-language-wrappers.ipynb).


## How to run the benchmarks

You can generate interactive notebooks by running
```julia
using Weave
notebook("multi-language-wrappers.jmd")
```
where you can replace `multi-language-wrappers.jmd` with the benchmark of your choice.

To generate the markdown files used in the documentation, we run
```julia
using Weave
weave("multi-language-wrappers.jmd"; doctype="github", out_path="../docs/src/benchmarks/")
```
