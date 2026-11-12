# Performance across versions

This page tracks how the cost of the solvers changes between versions of ProbNumDiffEq.jl.
It complements the [benchmarks](benchmarks/multi-language-wrappers.md), which compare solvers by accuracy per cost.
Here the computation stays the same and the version of the package changes.

Each benchmark is one full fixed-step `solve` of a test problem, with the same number of steps in every version.
The report shows the time, the allocations and the memory of each solve.

- **Check a change** compares two revisions on the current Julia version, by default the development version against the latest release.
- **History** shows every benchmark across all versions since v0.11.2.

**[Open the performance report](../performance_report.html)**

## How the numbers are made

All versions are measured on the same machine, with one Julia thread and one BLAS thread.
The report is updated by hand, usually after a release, so the development version in it can be older than the current `main` branch.
Releases that no longer install on the current Julia version are measured on Julia 1.10.

The suite is `performance/benchmarks.jl` in the repository, and `performance/README.md` describes how to run it.
