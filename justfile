default:
    just --list

format:
    julia --project=@JuliaFormatter -e 'using Pkg; Pkg.add("JuliaFormatter"); Pkg.update(); using JuliaFormatter; format(".")'

docs:
    julia --project=docs docs/make.jl

servedocs:
    julia --project=docs -e 'using LiveServer; serve(dir="docs/build")'

servedocs-continuously:
    julia --project=docs -e 'using ProbNumDiffEq, LiveServer; servedocs()'

benchmark:
    julia --project=benchmarks -e 'include("benchmarks/runall.jl")'

vale:
    git ls-files | xargs vale

# Performance benchmarks with AirspeedVelocity.jl, see performance/README.md
# Julia versions are juliaup channels; results go to performance/results/julia-<exact version>.

# The settings are exported, so that an override such as `just perf_threads=8 perf` also reaches the
# recipes that this file calls through `just`.
export perf_threads := env_var_or_default("perf_threads", "1")
# Julia for `main`, the working tree and every release that still runs on it
export perf_julia := env_var_or_default("perf_julia", "release")
export perf_releases := env_var_or_default("perf_releases", "v0.16.5 v0.16.6 v0.17.1 v0.18.0")
# Older releases that only run on an older Julia; v0.16.5 and v0.16.6 run on both and link the two
export perf_old_julia := env_var_or_default("perf_old_julia", "1.10")
export perf_old_releases := env_var_or_default("perf_old_releases", "v0.11.2 v0.12.1 v0.13.1 v0.14.0 v0.15.0 v0.16.5 v0.16.6")

export JULIA_NUM_PRECOMPILE_TASKS := "16"

# Install AirspeedVelocity into the environments @airspeedvelocity-<channel>
perf-install:
    for c in {{ perf_julia }} {{ perf_old_julia }}; do julia +$c --project=@airspeedvelocity-$c -e 'using Pkg; Pkg.add("AirspeedVelocity")'; done

# Compare the working tree against the local `main` branch
perf:
    {{ just_executable() }} _perf-run {{ perf_julia }} main
    PNDE_BENCH_STRICT=1 {{ just_executable() }} _perf-run {{ perf_julia }} dirty
    {{ just_executable() }} perf-table main,dirty
    {{ just_executable() }} perf-report

# Benchmark the releases that have no results from the current suite yet, on both Julia versions, and rerun `main`
perf-history:
    #!/usr/bin/env bash
    set -euo pipefail
    suite=$(git hash-object performance/benchmarks.jl | cut -c1-12)
    failed=""
    run() {
        dir=$({{ just_executable() }} _perf-dir "$1")
        if [ -f "$dir/results_ProbNumDiffEq@$2.json" ] && grep -qs "^suite = \"$suite\"" "$dir/meta_ProbNumDiffEq@$2.toml"; then return; fi
        {{ just_executable() }} _perf-run "$1" "$2" || failed="$failed $2@julia-$1"
    }
    for rev in {{ perf_releases }}; do run {{ perf_julia }} "$rev"; done
    for rev in {{ perf_old_releases }}; do run {{ perf_old_julia }} "$rev"; done
    {{ just_executable() }} _perf-run {{ perf_julia }} main
    {{ just_executable() }} perf-report
    if [ -n "$failed" ]; then echo "Benchmarking failed for:$failed" >&2; exit 1; fi

# Print the table of revisions that already have results, e.g. `just perf-table v0.18.0,main`
perf-table REVS JULIA=perf_julia:
    #!/usr/bin/env bash
    set -euo pipefail
    ratio=""
    if [ "$(echo "{{ REVS }}" | tr ',' '\n' | wc -l)" -eq 2 ]; then ratio="--ratio"; fi
    JULIAUP_CHANNEL={{ JULIA }} julia --startup-file=no --project=@airspeedvelocity-{{ JULIA }} \
        -e 'using AirspeedVelocity.BenchPkgTable: BenchPkgTable; exit(BenchPkgTable.command_main())' -- \
        ProbNumDiffEq --rev "{{ REVS }}" --input-dir "$({{ just_executable() }} _perf-dir {{ JULIA }})" --mode time,memory $ratio

# Build performance/results/report.html, a page for browsing all results
perf-report:
    julia --startup-file=no performance/report.jl

# Copy the report, without the working tree, to docs/src/performance_report.html, which the docs show
perf-publish:
    julia --startup-file=no performance/report.jl --publish

# Time to first solve in fresh processes, e.g. `just perf-ttfs v0.18.0 main dirty`
perf-ttfs +REVS:
    JULIAUP_CHANNEL={{ perf_julia }} julia --startup-file=no performance/ttfs.jl --exeflags "-t {{ perf_threads }} --heap-size-hint=32G" {{ REVS }}
    {{ just_executable() }} perf-report

# Benchmark one revision on a Julia channel, and record next to its results what was measured:
# the commit, the version of the suite, and the machine
_perf-run JULIA REV:
    #!/usr/bin/env bash
    set -euo pipefail
    dir=$({{ just_executable() }} _perf-dir {{ JULIA }})
    mkdir -p "$dir"
    if [ "{{ REV }}" = dirty ]; then
        commit="$(git rev-parse --short HEAD) + changes"; date=$(date +%F)
    else
        commit=$(git rev-parse --short "{{ REV }}^{commit}"); date=$(git log -1 --format=%cs "{{ REV }}")
    fi
    suite=$(git hash-object performance/benchmarks.jl | cut -c1-12)
    {{ just_executable() }} _perf-pkg {{ JULIA }} --rev "{{ REV }}"
    cpu=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)
    printf 'commit = "%s"\ndate = "%s"\nsuite = "%s"\nthreads = "%s"\ncpu = "%s"\nmeasured = "%s"\n' \
        "$commit" "$date" "$suite" "{{ perf_threads }}" "$cpu" "$(date -Is)" > "$dir/meta_ProbNumDiffEq@$(echo "{{ REV }}" | tr / _).toml"

# benchpkg on a Julia channel; its benchmark processes inherit the channel through JULIAUP_CHANNEL
_perf-pkg JULIA +ARGS:
    JULIAUP_CHANNEL={{ JULIA }} julia --startup-file=no --project=@airspeedvelocity-{{ JULIA }} \
        -e 'using AirspeedVelocity.BenchPkg: BenchPkg; exit(BenchPkg.command_main())' -- \
        --script performance/benchmarks.jl --output-dir "$({{ just_executable() }} _perf-dir {{ JULIA }})" \
        --exeflags='-t {{ perf_threads }} --heap-size-hint=32G' {{ ARGS }}

_perf-dir JULIA:
    @echo performance/results/julia-$(julia +{{ JULIA }} --startup-file=no -e 'print(VERSION)')
