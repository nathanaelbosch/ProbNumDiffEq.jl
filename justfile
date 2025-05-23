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
