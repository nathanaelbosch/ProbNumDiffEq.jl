# Copied from DiffEqDevTools.jl and modified
# See also: https://devdocs.sciml.ai/dev/alg_dev/test_problems/
DiffEqDevTools.appxtrue(sol::ProbODESolution, sol2::TestSolution) =
    DiffEqDevTools.appxtrue(mean(sol), sol2)
