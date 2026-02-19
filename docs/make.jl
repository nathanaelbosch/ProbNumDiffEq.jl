using Documenter
using DocumenterVitepress
using ProbNumDiffEq

using DocumenterCitations, Bibliography

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib"),
    style=:numeric,
)
sort_bibliography!(bib.entries, :nyt)  # name-year-title

makedocs(
    plugins=[bib],
    sitename="ProbNumDiffEq.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo="https://github.com/nathanaelbosch/ProbNumDiffEq.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="nathanaelbosch.github.io/ProbNumDiffEq.jl",
    ),
    modules=[ProbNumDiffEq],
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Second Order ODEs and Energy Preservation" => "tutorials/dynamical_odes.md",
            "Differential Algebraic Equations" => "tutorials/dae.md",
            "Probabilistic Exponential Integrators" => "tutorials/exponential_integrators.md",
            "Parameter Inference" => "tutorials/ode_parameter_inference.md",
        ],
        "Solvers and Options" => [
            "solvers.md",
            "priors.md",
            "initialization.md",
            "diffusions.md",
        ],
        "Data Likelihoods" => "likelihoods.md",
        "Benchmarks" => [
            "Multi-Language Wrapper Benchmark" => "benchmarks/multi-language-wrappers.md",
            "Non-stiff ODEs" => [
                "Lotka-Volterra" => "benchmarks/lotkavolterra.md",
                "Hodgkin-Huxley" => "benchmarks/hodgkinhuxley.md",
            ],
            "Stiff ODEs" => [
                "Van der Pol" => "benchmarks/vanderpol.md",
            ],
            "Second-order ODEs" => [
                "Pleiades" => "benchmarks/pleiades.md",
            ],
            "Differential-Algebraic Equations (DAEs)" => [
                "OREGO" => "benchmarks/orego.md",
                "ROBER" => "benchmarks/rober.md",
            ],
        ],
        "Internals" => [
            "Filtering and Smoothing" => "filtering.md",
            "Implementation via OrdinaryDiffEq.jl" => "implementation.md",
        ],
        "References" => "references.md",
    ],
    warnonly=Documenter.except(:missing_docs),
    checkdocs=:none,
)

DocumenterVitepress.deploydocs(
    repo="github.com/nathanaelbosch/ProbNumDiffEq.jl.git",
    devbranch="main",
    push_preview=true,
)
