using Documenter
using ProbNumODE

makedocs(
    sitename = "ProbNumODE.jl",
    format = Documenter.HTML(),
    modules = [ProbNumODE],
    pages = [
        "Home" => "index.md",
        "Examples" => [
            # "DifferentialEquations.jl Compatibility" => "diffeq_compatibility.md",
            # "Uncertainty Propagation" => "uncertain_startingvalues.md",
            "Uncertainty Quantification" => "uncertainty_quantification.md",
        ],
        "Internals" => "internals.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
