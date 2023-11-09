module DiffEqDevToolsExt

using ProbNumDiffEq
using DiffEqDevTools
using SciMLBase
using Statistics
using LinearAlgebra

function chi2(gaussian_estimate, actual_value)
    μ, Σ = gaussian_estimate
    diff = μ - actual_value
    if iszero(Σ)
        if iszero(diff)
	          return 1
        else
            throw(SingularException())
        end
    end
    chi2_pinv = diff' * (Σ \ diff)
    return chi2_pinv
end

function DiffEqDevTools.appxtrue(sol::ProbNumDiffEq.ProbODESolution, ref::TestSolution; kwargs...)
    ref.dense = sol.dense
    out = DiffEqDevTools.appxtrue(mean(sol), ref; kwargs...)
    out = _add_prob_errors!(out, sol, ref.interp)
    return out
end

function DiffEqDevTools.appxtrue(sol::ProbNumDiffEq.ProbODESolution, ref::SciMLBase.AbstractODESolution; kwargs...)
    out = DiffEqDevTools.appxtrue(mean(sol), ref; dense_errors=sol.dense, kwargs...)
    out = _add_prob_errors!(out, sol, ref)
    return out
end

function _add_prob_errors!(out, sol, ref)
    out.errors[:chi2_final] = chi2(sol.pu[end], ref.u[end])[1]
    if :l2 in keys(out.errors)
        out.errors[:chi2_steps] = mean(chi2.(sol.pu, ref.(sol.t)))
    end
    if :L2 in keys(out.errors)
        densetimes = collect(range(sol.t[1], stop=sol.t[end], length=100))
        interp_pu = sol(densetimes).u
        interp_ref = ref(densetimes).u
        out.errors[:chi2_interp] = mean(chi2.(interp_pu, interp_ref))
    end
    return out
end

end
