default:
    just --list

format:
    julia -e 'using JuliaFormatter; format(".")'

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

test-lower-compat:
    julia .just/downgrade.jl "Pkg,TOML" "v0"
    julia --project=. -e "using ProbNumDiffEq"
    git restore Project.toml