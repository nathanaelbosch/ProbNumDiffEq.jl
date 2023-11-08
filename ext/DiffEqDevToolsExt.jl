module DiffEqDevToolsExt

using ProbNumDiffEq
using DiffEqDevTools
using SciMLBase
using Statistics

# Copied from DiffEqDevTools.jl and modified
# See also: https://devdocs.sciml.ai/dev/alg_dev/test_problems/

DiffEqDevTools.appxtrue(sol::ProbNumDiffEq.ProbODESolution, sol2::TestSolution; kwargs...) =
    __appxtrue(sol, sol2; kwargs...)
DiffEqDevTools.appxtrue(
    sol::ProbNumDiffEq.ProbODESolution,
    sol2::SciMLBase.AbstractODESolution;
    kwargs...,
) = __appxtrue(sol, sol2; kwargs...)

function __appxtrue(sol, sol2; kwargs...)
    return DiffEqDevTools.appxtrue(mean(sol), sol2; dense_errors=sol.dense, kwargs...)
end

end
