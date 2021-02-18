using Documenter
using ProbNumDiffEq

makedocs(
    sitename = "ProbNumDiffEq.jl",
    format = Documenter.HTML(),
    modules = [ProbNumDiffEq],
    pages = [
        "Home" => "index.md",
        "Solvers and Options" => "solvers.md",
        # "Examples" => [
        #     "Comparison to ProbInts" => "probints_comparison.md"
        # ],
        # "Internals" => "internals.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/nathanaelbosch/ProbNumDiffEq.jl.git",
    devbranch="main"
)
