"""Compute the derivative df/dt(y,t), making use of dy/dt=f(y,t)"""
function get_derivative(f, d)
    dfdy(y, t) = d == 1 ?
        ForwardDiff.derivative((y) -> f(y, t), y) :
        ForwardDiff.jacobian((y) -> f(y, t), y)
    dfdt(y, t) = ForwardDiff.derivative((t) -> f(y, t), t)
    df(y, t) = dfdy(y, t) * f(y, t) + dfdt(y, t)
    return df
end

"""Compute q derivatives of f; Output includes f itself"""
function get_derivatives(f, d, q)
    out = Any[f]
    if q > 1
        for order in 2:q
            push!(out, get_derivative(out[end], d))
        end
    end
    return out
end
