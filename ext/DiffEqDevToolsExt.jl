module DiffEqDevToolsExt

using ProbNumDiffEq
using DiffEqDevTools
using SciMLBase
using Statistics
using LinearAlgebra

function chi2(gaussian_estimate, actual_value)
    μ, Σ = gaussian_estimate
    d = length(μ)
    diff = μ - actual_value
    if iszero(Σ)
        if iszero(diff)
            return one(eltype(actual_value))
        else
            @warn "Singular covariance matrix leads to bad (infinite) chi2 estimate"
            return convert(eltype(actual_value), Inf)
        end
    end
    @static if VERSION < v"1.7"
        return diff' * (Matrix(Σ) \ diff)
    else
        return diff' * (Σ \ diff)
    end
end

function DiffEqDevTools.appxtrue(
    sol::ProbNumDiffEq.ProbODESolution,
    ref::TestSolution;
    kwargs...,
)
    ref.dense = sol.dense
    out = DiffEqDevTools.appxtrue(mean(sol), ref; kwargs...)
    out = _add_prob_errors!(out, sol, ref.interp)
    return out
end

function DiffEqDevTools.appxtrue(
    sol::ProbNumDiffEq.ProbODESolution,
    ref::SciMLBase.AbstractODESolution;
    kwargs...,
)
    out = DiffEqDevTools.appxtrue(mean(sol), ref; dense_errors=sol.dense, kwargs...)
    out = _add_prob_errors!(out, sol, ref)
    return out
end

function _add_prob_errors!(out, sol, ref)
    out.errors[:chi2_final] = chi2(sol.pu[end], ref.u[end])[1]
    return out
end

end
