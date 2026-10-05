# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

ProbNumDiffEq.jl provides probabilistic numerical ODE solvers ("ODE filters": `EK0`, `EK1`, `DiagonalEK1`; `ExpEK`/`RosenbrockExpEK` are constructors for `EK0`/`EK1` with an `IOUP` prior) for the SciML/DifferentialEquations.jl ecosystem. They solve ODEs by Gaussian filtering and smoothing, so they return a posterior distribution instead of only a point estimate.

## Commands

Tests are split into groups via the `GROUP` env var: `Core`, `Solvers`, `Interface`, `CodeQuality` (Aqua, JET, ExplicitImports), `Downstream` (= Solvers + Interface), or `All` (default). CI runs each group separately.

```sh
julia --project -e 'using Pkg; Pkg.test()'                  # all tests
GROUP=Core julia --project -e 'using Pkg; Pkg.test()'       # one group
```

Each test file is standalone (brings its own `using` statements), so a single file can be run in the test environment. `test/Project.toml` points to the local package through `[sources]`, so instantiate it once (creates a gitignored `test/Manifest.toml`):

```sh
julia --project=test -e 'using Pkg; Pkg.instantiate()'  # once
julia --project=test test/core/filtering.jl             # one file
```

The test environment requires Julia ≥ 1.12, while the package itself supports Julia ≥ 1.10.

Other tasks (see `justfile`):

```sh
just format      # JuliaFormatter with .JuliaFormatter.toml; CI (FormatCheck) fails on any diff
just docs        # build docs: julia --project=docs docs/make.jl
just vale        # prose lint of Markdown/.jmd files (styles in .github/vale-styles)
just benchmark   # weave benchmarks/*.jmd into docs/src/benchmarks (slow)
```

CI also runs doctests: `julia --project=docs -e 'using Documenter: doctest; using ProbNumDiffEq; doctest(ProbNumDiffEq)'`.

## Architecture

The solvers are `OrdinaryDiffEqCore.OrdinaryDiffEqAdaptiveAlgorithm`s and reuse OrdinaryDiffEq's integrator loop, step-size control and Jacobian machinery by overloading its hooks; `docs/src/implementation.md` maps each hook to its file.

The package supports two major versions of several SciML packages at once (OrdinaryDiffEqCore 3/4, OrdinaryDiffEqDifferentiation 2/3, DiffEqBase 6/7, SciMLBase 2/3). Differences between them are bridged with `@static if isdefined(OrdinaryDiffEqCore, :name)` shims (for example `_set_EEst!`/`_get_EEst` in `perform_step.jl`, `_process_AD_choice` in `algorithms.jl`). Keep both branches working when editing them.

### State representation

- The filter state has dimension `D = d*(q+1)` (`d` = ODE dimension, `q` = prior order, i.e. number of modeled derivatives). It is ordered derivative-major: the first `d` entries are `u`, the next `d` are `u'`, and so on. `Proj(i)` / `cache.E0`, `E1`, `E2` extract the i-th derivative.
- Gaussians (`src/gaussians.jl`) hold covariances in square-root form as `PSDMatrices.PSDMatrix` (Σ = RᵀR). Predict, update and smooth (`src/filtering/`) propagate the factor `R` via QR and never form Σ explicitly.
- Every place that conditions the state on an observation (the ODE step, `DataUpdateCallback` and Fenrir, `ManifoldUpdate`, the initializations) builds a `LinearizedObservation(m, z, H, R)` (`src/filtering/update.jl`): an observation `y = h(x) + v`, `v ~ N(0, R)`, with `h(x) - y ≈ z + H (x - m)` linearized at `m`. `update!` conditions on it and is `update_mean!` followed by `update_cov!`, so that an iterated update (`ManifoldUpdate`) computes the covariance only for its last linearization. In the ODE step, the local diffusion and the error estimate share `H Q Hᵀ` from `observed_process_noise!`, and the global diffusion reads `z` and `S` from the observation and the update.
- Priors (`src/priors/`: `IWP`, `IOUP`, `Matern`) are Gauss–Markov processes discretized into a transition `Ah, Qh` for step `h`. `make_transition_matrices!` also computes preconditioners `P, PI` and the preconditioned `A = P Ah PI`, `Q = P Qh Pᵀ`. The filtering step uses `Ah, Qh`; dense output and sampling use the preconditioned `A, Q` for numerical stability.
- Diffusion models (`src/diffusions/`) are static (`FixedDiffusion`, `FixedMVDiffusion`: calibrated once after the solve) or dynamic (`DynamicDiffusion`, `DynamicMVDiffusion`: estimated in every step). The multivariate (`MV`) ones estimate one diffusion per ODE dimension and need block-diagonal covariances (see below).

### Covariance factorizations

The same filter code runs on three matrix representations, chosen by `covariance_structure(Alg, prior, diffusionmodel)` in `src/algorithms.jl`:

| `CovarianceStructure` | Matrix type | Used for |
|---|---|---|
| `IsometricKroneckerCovariance` | `IsometricKroneckerProduct` (`B ⊗ I_d`, `src/kronecker.jl`) | `EK0` with `IWP` prior and scalar diffusion |
| `BlockDiagonalCovariance` | `BlocksOfDiagonals` (`src/blocksofdiagonals.jl`) | `DiagonalEK1`, or `EK0` with multivariate diffusion |
| `DenseCovariance` | `Matrix` | `EK1`, and `EK0` with a non-IWP prior |

All cache matrices are created through `factorized_similar` / `factorized_zeros` / `to_factorized_matrix` (`src/covariance_structure.jl`), so their types follow the chosen structure. Any new linear-algebra operation used in the solve (predict, update, smoothing, marginalization, `diag!`, `_matmul!`, …) therefore needs a method for each of the three types. Plain linear algebra (matrix products, sums, the Kronecker vec trick, as used for the means in `predict_mean!` and `marginalize_mean!`) is overloaded on the structured types in `src/kronecker.jl` and `src/blocksofdiagonals.jl`. Filtering steps, which need QR, Cholesky or views into caches, are written once as a dense method; the Kronecker method calls it on the Kronecker factors through `on_kronecker_factors`, and the block-diagonal method calls it once per block through `foreach_diagonal_block` (`src/filtering/structured_covariances.jl`). Structured results are tested against dense ones in `test/core/`. Dense output and sampling must also keep the structure.

### Performance conventions

- Avoid allocations in `perform_step!`: temporaries are preallocated in `EKCache` and named by shape (`C_d`, `C_dxd`, `C_Dxd`, `C_DxD`, `C_2DxD`, `C_3DxD`).
- Use `_matmul!` (`src/fast_linalg.jl`) instead of `mul!`. It dispatches to Octavian.jl for BLAS-type dense matrices and to FastBroadcast (`@..`) for diagonal and scalar factors.
- `test/complexity.jl` asserts that the Kronecker `EK0` scales linearly in `d`; `test/core/fast_linalg.jl` covers `_matmul!`.

## Conventions enforced by tests

- **Explicit imports** (CodeQuality group, `test/explicit_imports.jl`): every name is imported explicitly and from its owner module, not through a re-exporting package. Deliberate uses of non-public API stay module-qualified and must be listed in `NONPUBLIC_ACCESSES` there; package internals used by `ext/` go in `OWN_INTERNALS_USED_BY_EXTENSIONS`.
- **Aqua.jl and JET.jl** run in the same group (`JET.test_package` with `target_modules=(ProbNumDiffEq,)`).
- Citations in docstrings use explicit display text, `[Author (Year)](@cite key)`, with keys from `docs/src/refs.bib`, so REPL help stays readable.

## Style

- Keep code comments rare. Add one only when the code cannot say it: a non-obvious invariant, or the formula behind terse in-place linear algebra.
- PRs are squash-merged and their description is taken from the commit message, so write commit messages in Markdown with code in backticks.
