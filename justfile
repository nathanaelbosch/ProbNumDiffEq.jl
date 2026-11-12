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

# Performance benchmarks with AirspeedVelocity.jl, see benchmark/README.md
# Julia versions are juliaup channels; results go to benchmark/results/julia-<exact version>.

bench_threads := "1"
# Julia for `main`, the working tree and every release that still runs on it
bench_julia := "release"
bench_releases := "v0.16.5 v0.16.6 v0.17.1 v0.18.0"
# Older releases that only run on an older Julia; v0.16.5 and v0.16.6 run on both and link the two
bench_old_julia := "1.10"
bench_old_releases := "v0.11.2 v0.12.1 v0.13.1 v0.14.0 v0.15.0 v0.16.5 v0.16.6"

export JULIA_NUM_PRECOMPILE_TASKS := "16"

# Install AirspeedVelocity into the environments @airspeedvelocity-<channel>
bench-install:
    for c in {{ bench_julia }} {{ bench_old_julia }}; do julia +$c --project=@airspeedvelocity-$c -e 'using Pkg; Pkg.add("AirspeedVelocity")'; done

# Compare the working tree against the local `main` branch
bench: (_bench-log bench_julia "main,dirty")
    {{ just_executable() }} _benchpkg {{ bench_julia }} --rev main
    PNDE_BENCH_STRICT=1 {{ just_executable() }} _benchpkg {{ bench_julia }} --rev dirty
    {{ just_executable() }} bench-table main,dirty
    {{ just_executable() }} bench-report

# Benchmark the releases that have no results yet on both Julia versions, and rerun `main`
bench-history:
    #!/usr/bin/env bash
    set -euo pipefail
    run() {
        dir=$({{ just_executable() }} _bench-dir "$1")
        if [ ! -f "$dir/results_ProbNumDiffEq@$2.json" ]; then
            {{ just_executable() }} _bench-log "$1" "$2"
            {{ just_executable() }} _benchpkg "$1" --rev "$2" || echo "Benchmarking $2 on Julia $1 failed" >&2
        fi
    }
    for rev in {{ bench_releases }}; do run {{ bench_julia }} "$rev"; done
    for rev in {{ bench_old_releases }}; do run {{ bench_old_julia }} "$rev"; done
    {{ just_executable() }} _bench-log {{ bench_julia }} main
    {{ just_executable() }} _benchpkg {{ bench_julia }} --rev main
    {{ just_executable() }} bench-report

# Print the table of revisions that already have results, e.g. `just bench-table v0.18.0,main`
bench-table REVS JULIA=bench_julia:
    #!/usr/bin/env bash
    set -euo pipefail
    ratio=""
    if [ "$(echo "{{ REVS }}" | tr ',' '\n' | wc -l)" -eq 2 ]; then ratio="--ratio"; fi
    JULIAUP_CHANNEL={{ JULIA }} julia --startup-file=no --project=@airspeedvelocity-{{ JULIA }} \
        -e 'using AirspeedVelocity.BenchPkgTable: BenchPkgTable; exit(BenchPkgTable.command_main())' -- \
        ProbNumDiffEq --rev "{{ REVS }}" --input-dir "$({{ just_executable() }} _bench-dir {{ JULIA }})" --mode time,memory $ratio

# Build benchmark/results/report.html, a page for browsing all results
bench-report:
    julia --startup-file=no benchmark/report.jl

# Copy the report, without the working tree, to docs/src/performance_report.html, which the docs show
bench-publish:
    julia --startup-file=no benchmark/report.jl --publish

# Time to first solve in fresh processes, e.g. `just bench-ttfs v0.18.0 main dirty`
bench-ttfs +REVS: (_bench-log bench_julia "ttfs " + REVS)
    JULIAUP_CHANNEL={{ bench_julia }} julia --startup-file=no benchmark/ttfs.jl --exeflags "-t {{ bench_threads }} --heap-size-hint=32G" {{ REVS }}
    {{ just_executable() }} bench-report

# benchpkg on a Julia channel; its benchmark processes inherit the channel through JULIAUP_CHANNEL
_benchpkg JULIA +ARGS:
    JULIAUP_CHANNEL={{ JULIA }} julia --startup-file=no --project=@airspeedvelocity-{{ JULIA }} \
        -e 'using AirspeedVelocity.BenchPkg: BenchPkg; exit(BenchPkg.command_main())' -- \
        --script benchmark/benchmarks.jl --output-dir "$({{ just_executable() }} _bench-dir {{ JULIA }})" \
        --exeflags='-t {{ bench_threads }} --heap-size-hint=32G' {{ ARGS }}

_bench-dir JULIA:
    @echo benchmark/results/julia-$(julia +{{ JULIA }} --startup-file=no -e 'print(VERSION)')

_bench-log JULIA REVS:
    #!/usr/bin/env bash
    dir=$({{ just_executable() }} _bench-dir {{ JULIA }})
    mkdir -p $dir
    cpu="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
    echo "$(date -Is) revs={{ REVS }} host=$(hostname) cpu=$cpu $(julia +{{ JULIA }} --version) threads={{ bench_threads }}" >> $dir/runs.log
