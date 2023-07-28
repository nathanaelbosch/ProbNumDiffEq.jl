using Documenter, DocumenterCitations
using ProbNumDiffEq

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib"),
    # style=:authoryear,
    style=:numeric,
)

makedocs(
    bib,
    sitename="ProbNumDiffEq.jl",
    format=Documenter.HTML(
        assets=String["assets/citations.css"],
    ),
    modules=[ProbNumDiffEq],
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md"
            "Second Order ODEs and Energy Preservation" => "tutorials/dynamical_odes.md"
            "Differential Algebraic Equations" => "tutorials/dae.md"
            "Probabilistic Exponential Integrators" => "tutorials/exponential_integrators.md"
            "Parameter Inference" => "tutorials/fenrir.md"
        ],
        "Solvers and Options" => [
            "solvers.md",
            "priors.md",
            "initialization.md",
            "diffusions.md",
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
