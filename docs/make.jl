using Documenter
using ODEFilters

makedocs(
    sitename = "ODEFilters.jl",
    format = Documenter.HTML(),
    modules = [ODEFilters],
    pages = [
        "Home" => "index.md",
        "Solvers and Options" => "manual.md",
        "Examples" => [
            "Comparison to ProbInts" => "probints_comparison.md"
        ]
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nathanaelbosch/ODEFilters.jl.git",
    devbranch="main"
)
