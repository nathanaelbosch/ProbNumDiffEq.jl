"""
    modelingtoolkitize_with_jac(prob::ODEProblem)

Add a jacobian function to the ODE function, using ModelingToolkit.jl.
This function is also used to internally transform out-of-place problems to in-place.
"""
function modelingtoolkitize_with_jac(prob::ODEProblem; jac=true)
    if !isnothing(prob.f.analytic)
        error("`modelingtoolkitize_with_jac` does not preserve analytic solutions")
    end
    return ODEProblem(modelingtoolkitize(prob), prob.u0, prob.tspan, jac=jac)
end
