using Documenter
using ProbNumDiffEq

using DocumenterCitations, Bibliography

# DocumenterCitations.bib_sorting(::Val{:numeric}) = :nyt
# function DocumenterCitations.format_bibliography_label(
#     ::Val{:numeric},
#     entry,
#     citations::DocumenterCitations.OrderedDict{String,Int64},
# )
#     key = entry.id
#     sorted_bibtex_keys = citations |> keys |> collect |> sort
#     i = findfirst(x -> x == key, sorted_bibtex_keys)
#     @info "format_bibliography_label" entry.id citations sorted_bibtex_keys i entry.date
#     return "[$i]"
# end
# Bibliography.sorting_rules[:nyt] = [:authors; :date; :title]

bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib"),
    style=:numeric,
    # style=:authoryear,
)
sort_bibliography!(bib.entries, :nyt)  # name-year-title

makedocs(
    plugins=[bib],
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
        ],
        "Solvers and Options" => [
            "solvers.md",
            "priors.md",
            "initialization.md",
            "diffusions.md",
        ],
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
            "Filtering and Smoothing" => "filtering.md"
            "Implementation via OrdinaryDiffEq.jl" => "implementation.md"
        ],
        "References" => "references.md",
    ],
    warnonly=Documenter.except(:missing_docs),
    checkdocs=:none,
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/nathanaelbosch/ProbNumDiffEq.jl.git",
    devbranch="main",
    push_preview=true,
)
