"""initialize x0 up to the provided order"""
function initial_update!(integ, cache, init::TaylorModeInit)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = cache
    q = integ.alg.order

    @unpack x_tmp, x_tmp2, m_tmp, K1, K2 = cache

    f_derivatives = taylormode_get_derivatives(u, f, p, t, q)
    @assert length(0:q) == length(f_derivatives)
    for (o, df) in zip(0:q, f_derivatives)

        condition_on!(x, Proj(o), df, m_tmp, K1, K2, x_tmp.Σ, x_tmp2.Σ.mat)
    end
end


"""
    Compute initial derivatives of an OOP ODEProblem with TaylorSeries.jl

    Warning: The OOP version is much slower than the IIP version!
"""
function taylormode_get_derivatives(u, f::AbstractODEFunction{false}, p, t, q)
    d = length(u)

    f_vec = f_to_vector_valued(f, u)

    u_vec = u[:]

    # Simplify further:
    _f(u) = f_vec(u, p, t)

    u0 = u_vec
    du0 = _f(u_vec)
    if q == 1
        return [u0, du0]
    end

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(_t -> f_vec(u_vec, p, _t), t)
    @assert !(eltype(f_t_taylor) <: TaylorN) "The vector field depends on t; The code might not yet be able to handle these (but it should be easy to implement)"

    set_variables("u", numvars=d, order=q+1)

    fp = taylor_expand(_f, u_vec)
    f_derivatives = [fp]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu = stack([TaylorSeries.derivative.(_curr_f_deriv, i) for i in 1:d])'
        df = dfdu * fp
        push!(f_derivatives, df)
    end

    return [u_vec, evaluate.(f_derivatives)...]
end
"""
    Compute initial derivatives of an IIP ODEProblem with TaylorIntegration.jl
"""
function taylormode_get_derivatives(u, f::AbstractODEFunction{true}, p, t, q)

    # f = f_to_vector_valued(f, u)
    # u_vec = u[:]

    tT = Taylor1(typeof(t), q)
    tT[0] = t
    uT = Taylor1.(u, tT.order)
    duT = zero.(Taylor1.(u, tT.order))
    uauxT = similar(uT)
    TaylorIntegration.jetcoeffs!(f, tT, uT, duT, uauxT, p)
    return [evaluate.(differentiate.(uT, i)) for i in 0:q]
end

"""
    Compute initial derivatives of a SecondOrderODE with TaylorSeries.jl
"""
function taylormode_get_derivatives(u::ArrayPartition, f::DynamicalODEFunction, p, t, q)

    d = length(u[1,:])
    Proj(deriv) = deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    f_oop(du, u, p, t) = !isinplace(f.f1) ? f.f1(du, u, p, t) :
        (ddu = copy(du); f.f1(ddu, du, u, p, t); return ddu)

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(_t -> f_oop(u[1:d], u[d+1:end], p, _t), t)
    @assert !(eltype(f_t_taylor) <: TaylorN) "The vector field depends on t; The code might not yet be able to handle these (but it should be easy to implement)"

    set_variables("u", numvars=2d, order=q+1)

    fp1 = taylor_expand(u -> f_oop(u[1:d], u[d+1:end], p, t), u[:])
    fp2 = taylor_expand(u -> u[1:d], u[:])
    f_derivatives = [fp1]
    for o in 3:q
        _curr_f_deriv = f_derivatives[end]
        dfdu1 = stack([derivative.(_curr_f_deriv, i) for i in 1:d])'
        dfdu2 = stack([derivative.(_curr_f_deriv, i) for i in d+1:2d])'
        df = dfdu1 * fp1 + dfdu2 * fp2
        push!(f_derivatives, df)
    end

    return [u[2,:], u[1,:], evaluate.(f_derivatives)...]
end
