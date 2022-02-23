
function DiffEqBase.__init(
    prob::DiffEqBase.AbstractODEProblem{uType, tType, false},
    alg::GaussianODEFilter, args...; kwargs...) where {uType, tType}
    @warn "The given problem is in out-of-place form, but since the algorithms in this package are written for in-place problems it will be automatically converted using ModelingToolkit.jl. For more control, specify the ODEProblem directly in in-place form."
    prob = modelingtoolkitize_with_jac(prob; jac=!isnothing(prob.f.jac))
    return DiffEqBase.__init(prob, alg, args...; kwargs...)
end
