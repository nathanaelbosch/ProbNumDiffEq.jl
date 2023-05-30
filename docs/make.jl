using Documenter
using ProbNumDiffEq

makedocs(
    sitename="ProbNumDiffEq.jl",
    format=Documenter.HTML(),
    modules=[ProbNumDiffEq],
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "getting_started.md"
            "Second Order ODEs and Energy Preservation" => "dynamical_odes.md"
            "Differential Algebraic Equations" => "dae.md"
            "Parameter Inference" => "fenrir.md"
            "Probabilistic Exponential Integrators" => "exponential_integrators.md"
        ],
        "Solvers and Options" => [
            "solvers.md",
            "diffusions.md",
            "initialization.md",
            "priors.md",
        ],
        "Benchmarks" => [
            "Multi-Language Wrapper Benchmark" => "benchmarks/multi-language-wrappers.md",
            "Non-stiff ODE: Lotka-Volterra" => "benchmarks/lotkavolterra.md",
            "Stiff ODE: Van der Pol" => "benchmarks/vanderpol.md",
            "DAE: ROBER" => "benchmarks/rober.md",
        ],
        "Internals" => [
            "Filtering and Smoothing" => "filtering.md"
            "Implementation via OrdinaryDiffEq.jl" => "implementation.md"
        ],
    ],
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/nathanaelbosch/ProbNumDiffEq.jl.git",
    devbranch="main",
    push_preview=true,
)
