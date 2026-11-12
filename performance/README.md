# Performance benchmarks

This directory holds a performance benchmark suite for [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl).
It measures the run time, allocations and memory of `solve` calls, and compares them between git revisions: the working tree against `main`, or releases with each other.

It is different from `benchmarks/` (plural), which holds the work-precision diagrams shown in the documentation.
Those compare solvers by accuracy per cost; this suite tracks how the cost of the same computation changes between versions of the package.

## Usage

The recipes are in the `justfile`:

```sh
just perf-install            # once: installs AirspeedVelocity into @airspeedvelocity-<channel> for both Julia channels
just perf                    # working tree against the local `main` branch
just perf-history            # all releases in `perf_releases` and `perf_old_releases` without results yet, plus `main`
just perf-table v0.18.0,main # table of revisions that already have results
just perf-ttfs main dirty    # time to first solve, see below
just perf-report             # rebuild performance/results/report.html
just perf-publish            # write docs/src/performance_report.html, the copy shown in the docs
```

Open `performance/results/report.html` in a browser to browse the results.
It is rebuilt after every run by `performance/report.jl`, from the template `performance/report_template.html`, and has two views.

- **Check a change** (on the current Julia): a headline with the geometric mean, every benchmark as a dot by its change, the list of what got slower, and a table of all benchmarks with time, allocations and memory for baseline → revision.
  Clicking a dot, a list entry or a row shows the charts of that benchmark.
  The changes by kind of benchmark are folded away below the list.
- **History**: one table of every benchmark across all versions.
  Every cell is compared with a baseline that you choose, the latest release by default, and each column head shows the overall change of that version against it.
  Older releases are shown as measured on the older Julia they run on.
  The release that runs on both Julia versions appears twice, side by side, which shows the effect of Julia alone.

The docs page "Performance across versions" (`docs/src/performance.md`) links to `docs/src/performance_report.html`, which opens as its own page and links back.
That file is committed, and is only updated by `just perf-publish`, typically after a release and a `just perf-history`.
It leaves out the working tree, and does not contain the host name of the machine, only its CPU model.
The docs build does not run the benchmarks: timings from CI machines are noisy and not comparable with stored results.

The report embeds at most 100 solves per benchmark and revision, together with the statistics of all solves, which keeps it at about 0.5 MB.

Times are compared by the fastest solve of each benchmark, not the median.
The timings come in clusters, since some solves are hit by garbage collection, and the median jumps between clusters from run to run: between v0.16.6 and v0.17.1 the median changed by up to 12% and the minimum by at most 5% (2026-11).

The summary by kind of benchmark groups the benchmarks by three factors: the covariance structure of the solver, the size of the problem (d ≤ 3 or d ≥ 14), and filter only against filter + smoother.
These factors, the step counts and the `solve` call shown for each benchmark are declared in `PROBLEMS` and `SOLVERS` in `performance/report.jl`; update them when `benchmarks.jl` changes.

Julia versions are given as juliaup channels: `perf_julia` (`release`) for `main`, the working tree and every release that runs on it, and `perf_old_julia` (`1.10`) for older releases.
Results are written to `performance/results/julia-<exact Julia version>/` (gitignored), one `results_ProbNumDiffEq@<rev>.json` per revision.
`benchpkgtable` only reads these files, so results from separate runs can be combined.
`just perf-history` reuses the results of releases and only reruns `main`.
When a channel moves to a new Julia version, its results start in a new folder, and `just perf-history` reruns the releases there.
`runs.log` in each folder records the host, CPU, Julia version and thread count of each run.

Results are only comparable when they come from the same machine, Julia version and thread settings.
Delete `performance/results/` when the machine or the thread settings change.

## What is measured

All benchmarks are fixed-step solves (`adaptive=false`), so that every version does the same number of steps and a difference in time is a difference in cost per step.
A change in the step-size control or in accuracy is therefore not visible here.
Each configuration runs twice: `filter` (`smooth=false`, `dense=false`) and `smoother` (`smooth=true`).
AirspeedVelocity also adds `time_to_load`, the time of `using ProbNumDiffEq`.

| Problem | d | Solvers |
|---|---|---|
| `lotkavolterra` | 2 | `EK0`, `EK1`, `DiagonalEK1`, `EK0` with `DynamicMVDiffusion` and `FixedMVDiffusion`, `EK1` with `IOUP` and `Matern` priors |
| `vanderpol` (μ = 10) | 2 | `EK1`, `ExpEK`, `RosenbrockExpEK` |
| `pleiades` | 28 | `EK0`, `EK1`, `DiagonalEK1` |
| `pleiades_secondorder` | 14 | `EK0`, `EK1` on the `SecondOrderODEProblem` |
| `lorenz96` | 50 | `EK0`, `EK1`, `DiagonalEK1`, `EK0` with `DynamicMVDiffusion`, `MatfreeEK1` |
| `rober_massmatrix` | 3 | `EK1` on Robertson's problem with a singular mass matrix |
| `lotkavolterra_data` | 2 | `EK0`, `EK1` with a `DataUpdateCallback` (100 data points) |
| `oscillator_manifold` | 2 | `EK1` with a `ManifoldUpdate` |

One version takes about 4 minutes: 1.5 for installing and precompiling, 2.5 for the suite.

## Time to first solve

`performance/ttfs.jl` measures what the suite cannot: BenchmarkTools runs every benchmark once before measuring, so the suite never sees compilation.
For each revision, it installs and precompiles the package, then starts 5 fresh Julia processes and times `using ProbNumDiffEq` and the first `solve` with `EK0` and with `EK1` on a small problem.
It prints the medians and writes `performance/results/julia-<version>/ttfs_ProbNumDiffEq@<rev>.toml`.

## Running the suite on old versions

The suite in the working tree is run on every revision (`--script`), so it may only use the public API that the oldest benchmarked release already has.
An entry whose trial solve throws, for example because a solver does not exist yet in that version, is skipped with a warning and has no value in the table.
`just perf` sets `PNDE_BENCH_STRICT=1` for the working tree, so that a benchmark broken by the change under test fails instead of being skipped.

Entries missing in releases, and why:

- `DiagonalEK1` before v0.15, `ExpEK` and `RosenbrockExpEK` before v0.12: these solvers did not exist yet.
- `lorenz96/MatfreeEK1`, in all releases up to v0.18.0: the `MatfreeEK1` is not released yet.
- `pleiades_secondorder`, in v0.12 and v0.13: the `SecondOrderODEProblem` fails with a size mismatch.
- `lotkavolterra_data`: the `DataUpdateCallback` does not exist before v0.14; in v0.14.0 the `EK0` fails with it, and in v0.15.0 to v0.16.6 (v0.16.6 only on Julia 1.13) the `EK0` returns `Unstable` and the `EK1` fails a Cholesky factorization.
- `rober_massmatrix`, in v0.15.0 to v0.16.6 (v0.16.6 only on Julia 1.13): DAE initialization with a singular mass matrix needed `OrdinaryDiffEqNonlinearSolve` to be loaded.

### Which Julia runs which release

Old releases only install with the old SciML dependencies they allow, and those do not all work on recent Julia versions (checked 2026-11):

| Release | Julia 1.10 | Julia 1.13 |
|---|---|---|
| v0.10 and older | fail to precompile | fail |
| v0.11.2 to v0.15.0 | run | fail to precompile |
| v0.16.0 to v0.16.4 | fail (also on 1.12) | fail |
| v0.16.5, v0.16.6 | run | run |
| v0.17.0 | not tried | installs, but fails to load |
| v0.17.1, v0.18.0 | not needed | run |

On Julia 1.13, v0.16.0 to v0.16.4 get an `OrdinaryDiffEqDifferentiation` that overwrites a method during precompilation, and v0.15.0 and older get a `LinearSolve` whose precompile workload fails with the current `KLU`.
v0.17.0 admits an `OrdinaryDiffEqDifferentiation` that no longer defines `standardtag`.

So `perf_releases` (Julia 1.13) lists v0.16.5 and later, except v0.17.0, and `perf_old_releases` (Julia 1.10) lists the latest patch of v0.11 to v0.15, plus v0.16.5 and v0.16.6, which link the two.

Julia and the dependencies resolved for it change the timings a lot: v0.16.6 is about 30% faster on Julia 1.13 than on 1.10, and allocates about 1.8× as many objects, since Julia 1.11 changed how arrays are allocated.
So "Check a change" only compares revisions on the current Julia.
In History, a comparison of an old release with a baseline on the current Julia includes the effect of Julia itself.
The headline of History separates the two by chaining steps: releases on Julia 1.10 against each other, one step that changes only Julia (v0.16.6 on both), then releases on Julia 1.13.
That step also shows what the dependencies resolved on Julia 1.10 do: with v0.16.5 and v0.16.6 the Kronecker `EK0` allocates about 100 MiB per solve there, against 1 MiB on Julia 1.13, on v0.15.0, and on `main` with Julia 1.10.

## Threads

Benchmarks run with one Julia thread and one BLAS thread.
Both change the timings: `_matmul!` uses Octavian, which multithreads larger products over Julia threads, and OpenBLAS multithreads the factorizations.
The Julia thread count is the `justfile` variable `perf_threads`, e.g. `just perf_threads=8 perf`; keep such results apart from single-threaded ones.
The BLAS thread count is fixed to one in `benchmarks.jl`.
