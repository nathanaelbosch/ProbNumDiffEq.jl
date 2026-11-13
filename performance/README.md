# Performance benchmarks

This suite tracks how the cost of the same computation changes between versions of the package.
`benchmarks/` (plural) is unrelated: it holds the work-precision diagrams of the documentation.

Each of the 48 benchmarks is one full fixed-step `solve` (`adaptive=false`), so every version does the same number of steps and a difference in time is a difference in cost per step.
Changes in step-size control or accuracy are therefore not visible here.
The suite runs with [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl), with one Julia thread and one BLAS thread.

## Usage

```sh
just perf install            # once: AirspeedVelocity for both Julia versions
just perf                    # the working tree against `main`, about 8 minutes
just perf run v0.18.0 main   # any revisions, measured in one session
just perf history            # the releases without current results, and `main`
just perf ttfs v0.18.0 main  # time to first solve, in fresh processes
just perf report             # rebuild performance/results/report.html
just perf report --publish   # rebuild docs/src/performance_report.html, the copy in the docs
```

The commands are in `performance/run.jl`; `just perf` only calls it, so `julia performance/run.jl history` works without `just`.

Open `performance/results/report.html` to browse the results.
The page explains itself under "How to read this".

## Results

Results go to `performance/results/julia-<version>/` (gitignored), one `results_*.json` per revision, with a `meta_*.toml` that records the commit, the version of `benchmarks.jl`, the CPU and the thread count.
`just perf history` reuses the results of a release until `benchmarks.jl` changes.

Results are only comparable from the same machine, Julia version and thread settings, and times are most reliable from the same session.
Two runs of the same commit a day apart differed by up to 11%, mostly in the same direction, while the allocations were identical (2026-11).
So measure both revisions in one session before acting on a change, as `just perf` does, e.g. `just perf run v0.18.0 main`.

Time is compared by the fastest solve, which is more repeatable than the median: between releases with nearly identical code, the minimum changed by at most 8% and the median by up to 13% (2026-11).

## Publishing

The docs menu entry "Performance across versions" opens `docs/src/performance_report.html`, a committed file that is only updated by `just perf report --publish`, usually after a release.
It leaves out the working tree and the host name.
The docs build does not run the suite, since timings on CI machines are not comparable.

## Old releases

The suite in the working tree runs on every revision, so it may only use public API.
An entry whose trial solve fails, for example because the solver did not exist yet, is skipped; the working tree runs with `PNDE_BENCH_STRICT=1` so that such failures throw instead.
Besides solvers and features that did not exist yet, the gaps in the report are `SecondOrderODEProblem` in v0.12 and v0.13, `DataUpdateCallback` in v0.14 to v0.16, and the singular mass matrix in v0.15 to v0.16, which failed in those releases.

Old releases only install with old SciML dependencies, which not all Julia versions can load (checked 2026-11):

| Release | Julia 1.10 | Julia 1.13 |
|---|---|---|
| v0.10 and older | fail | fail |
| v0.11.2 to v0.15.0 | run | fail |
| v0.16.0 to v0.16.4 | fail | fail |
| v0.16.5, v0.16.6 | run | run |
| v0.17.0 | not tried | fails to load |
| v0.17.1, v0.18.0 | not needed | run |

`RELEASES` and `OLD_RELEASES` in `performance/run.jl` list the releases for each Julia; add a new release to `RELEASES`.
Julia itself changes the timings: v0.16.6 is about 30% faster on Julia 1.13 than on 1.10 (2026-11).
The report shows this as a separate step, measured on v0.16.6 on both.

## Maintenance

The labels, step counts, `solve` calls and factors in the report come from `PROBLEMS` and `SOLVERS` in `performance/report.jl`; update them with `benchmarks.jl`.
