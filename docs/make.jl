using Documenter
using ProbNumDiffEq

makedocs(
    sitename="ProbNumDiffEq.jl",
    format=Documenter.HTML(),
    modules=[ProbNumDiffEq],
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Introduction to ODE Filters" => "getting_started.md"
            "Second Order ODEs and Energy Preservation" => "dynamical_odes.md"
            "Differential Algebraic Equations" => "dae.md"
        ],
        "Solvers and Options" => "solvers.md",
        "Benchmark" => "benchmark.md",
        "Internals" => [
            "Filtering and Smoothing" => "filtering.md"
            "Implementation via OrdinaryDiffEq.jl" => "implementation.md"
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(repo="github.com/nathanaelbosch/ProbNumDiffEq.jl.git", devbranch="main")
