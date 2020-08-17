"""Compute the derivative df/dt(y,t), making use of dy/dt=f(y,t)"""
function _get_derivative(f, d)
    dfdy(y, t) = d == 1 ?
        ForwardDiff.derivative((y) -> f(y, t), y) :
        ForwardDiff.jacobian((y) -> f(y, t), y)
    dfdt(y, t) = ForwardDiff.derivative((t) -> f(y, t), t)
    df(y, t) = dfdy(y, t) * f(y, t) + dfdt(y, t)
    return df
end

"""Compute q derivatives of f; Output includes f itself"""
function _get_derivatives(f, d, q)
    out = Any[f]
    if q > 1
        for order in 2:q
            push!(out, _get_derivative(out[end], d))
        end
    end
    return out
end


"""Compute the q derivatives of the rhs function for a given ODE problem"""
function get_initial_derivatives(prob, order)
    u0 = prob.u0
    d = length(u0)
    q = order
    f = prob.f
    t0 = prob.tspan[1]
    p = prob.p

    derivatives = _get_derivatives((x, t) -> f(x, p, t), d, q)
    m0 = vcat(u0, [_f(u0, t0) for _f in derivatives]...)
    return m0
end



"""Nice, but slower than the ForwardDiff approach"""
function _get_init_derivatives_mtk(prob, order)
    # Output of size order+1
    u0 = prob.u0
    d = length(u0)
    q = order
    out = fill(zero(u0[1]), d*(q+1))

    sys = modelingtoolkitize(prob)
    t = sys.iv()
    @derivatives D'~t
    u = [s(t) for s in sys.states]
    p = [_p() for _p in sys.ps]

    rhs = ModelingToolkit.rhss(sys.eqs)

    substitutions = [u .=> u0; p .=> prob.p; t .=> prob.tspan[1]]

    out[1:d] .= u0

    val = substitute.(rhs, (substitutions,))

    out[d+1:2d] .= [v.value for v in val]
    u = D.(u)
    substitutions = [D.(u) .=> val; substitutions]

    for i in 2:(order)
        rhs = ModelingToolkit.jacobian(rhs, [t])[:]
        val = expand_derivatives.(substitute.(rhs, (substitutions,)))
        out[(i*d)+1:(i+1)*d] .= [v.value for v in val]
        u = D.(u)
        substitutions = [u .=> val; substitutions]
    end
    return out
end
