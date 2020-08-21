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

    substitutions = [u .=> u0; t .=> prob.tspan[1]]
    if length(p) > 0
        substitutions = [substitutions; p .=> prob.p]
    end

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



function get_initial_states_forwarddiff(prob::ODEProblem, order::Int)
    f = isinplace(prob) ? iip_to_oop(prob.f) : prob.f

    u0 = prob.u0
    t0 = prob.tspan[1]
    p = prob.p

    d = length(u0)
    q = order

    m0 = zeros(d*(q+1))
    P0 = zeros(d*(q+1), d*(q+1))

    m0[1:d] .= u0
    m0[d+1:2d] .= f(u0, p, t0)

    f_derivatives = Function[f]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu(u, p, t) = ForwardDiff.jacobian(u -> _curr_f_deriv(u, p, t), u)
        dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        push!(f_derivatives, df)
        m0[o*d+1:(o+1)*d] = df(u0, p, t0)
    end

    return m0, P0
end


function iip_to_oop(f!)
    function f(u, p, t)
        du = copy(u)
        f!(du, u, p, t)
        return du
    end
    return f
end


function remake_prob_with_jac(prob)
    prob = remake(prob, p=collect(prob.p))
    sys = modelingtoolkitize(prob)
    jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
    f = ODEFunction(prob.f.f, jac=jac)
    return remake(prob, f=f)
end
