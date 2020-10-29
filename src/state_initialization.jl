function initialize_without_derivatives(u0, f, p, t0, order, var=1e-3)
    q = order
    d = length(u0)

    m0 = zeros(d*(q+1))
    m0[1:d] = u0
    if !isinplace(f)
        m0[d+1:2d] = f(u0, p, t0)
    else
        f(m0[d+1:2d], u0, p, t0)
    end
    P0 = [zeros(d, d) zeros(d, d*q);
          zeros(d*q, d) diagm(0 => var .* ones(d*q))]
    return m0, P0
end


function initialize_with_derivatives_forwarddiff(u0, f, p, t0, order::Int)
    @warn "Better to use the TaylorSeries.jl approach"
    f = isinplace(f) ? iip_to_oop(f) : f

    d = length(u0)
    q = order

    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

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


function initialize_with_derivatives(u0, f, p, t0, order::Int)
    taylor_order = 15
    @assert taylor_order > order "The internal order of TaylorSeries.jl is smaller than the requested order of derivatives"

    f = isinplace(f) ? iip_to_oop(f) : f

    d = length(u0)
    q = order

    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

    m0[1:d] .= u0
    m0[d+1:2d] .= f(u0, p, t0)

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(t -> f(u0, p, t), t0)
    @assert !(eltype(f_t_taylor) <: TaylorN)

    fp = taylor_expand(u -> f(u, p, t0), u0; order=taylor_order)
    f_derivatives = [fp]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        # dfdu(u, p, t) = ForwardDiff.jacobian(u -> _curr_f_deriv(u, p, t), u)
        dfdu = [derivative.(_curr_f_deriv, 1) derivative.(_curr_f_deriv, 2)]
        # dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        # df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        df = dfdu * fp
        push!(f_derivatives, df)
        m0[o*d+1:(o+1)*d] = evaluate(df)
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
