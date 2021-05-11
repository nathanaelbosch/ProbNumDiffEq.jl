"""initialize x0 up to the provided order"""
function initial_update!(integ)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = integ.cache
    q = integ.alg.order
    return initial_update!(x, u, f, p, t, q)
end
function initial_update!(x, u, f, p, t, q)
    d = length(u)
    # TODO: Find a proper place for `Proj` instead of duplicating it everywhere
    Proj(deriv) = deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    f_oop = isinplace(f) ? iip_to_oop(f) : f

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(_t -> f_oop(u, p, _t), t)
    @assert !(eltype(f_t_taylor) <: TaylorN) "The vector field depends on t; The code might not yet be able to handle these (but it should be easy to implement)"

    # Simplify further:
    _f(u) = f_oop(u, p, t)

    # Condition on Proj(0)*x = u0
    condition_on!(x, Proj(0), u)
    condition_on!(x, Proj(1), _f(u))

    set_variables("u", numvars=d, order=q+1)

    fp = taylor_expand(_f, u)
    f_derivatives = [fp]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu = stack([TaylorSeries.derivative.(_curr_f_deriv, i) for i in 1:d])'
        # dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        # df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        df = dfdu * fp
        push!(f_derivatives, df)
        condition_on!(x, Proj(o), evaluate(df))
    end

    return nothing
end

# TODO Either name texplicitly for the initial update, or think about how to use this in general
function condition_on!(x::SRGaussian, H::AbstractMatrix, data::AbstractVector)
    z = H*x.μ
    S = X_A_Xt(x.Σ, H)
    K = x.Σ * H' * inv(S)
    x.μ .+= K*(data - z)
    newcov = X_A_Xt(x.Σ, I-K*H)
    copy!(x.Σ, newcov)
    nothing
end


"""Quick and dirty wrapper to make IIP functions OOP"""
function iip_to_oop(f!)
    function f(u, p, t)
        du = copy(u)
        f!(du, u, p, t)
        return du
    end
    return f
end








function initial_update!(x, u::ArrayPartition, f::DynamicalODEFunction, p, t, q)
    _du, _u = u[1, :], u[2, :]
    stacked_u = [_du; _u]
    f.f1, f.f2

    d = length(_u)
    Proj(deriv) = deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    f_oop(du, u, p, t) = (ddu = copy(du); f.f1(ddu, du, u, p, t); return ddu)

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(_t -> f_oop(_du, _u, p, _t), t)
    @assert !(eltype(f_t_taylor) <: TaylorN) "The vector field depends on t; The code might not yet be able to handle these (but it should be easy to implement)"

    # Simplify further:
    _f(du, u) = f_oop(du, u, p, t)
    _f(stacked_u) = _f(stacked_u[1:d], stacked_u[d+1:end])

    # Condition on Proj(0)*x = u0
    condition_on!(x, Proj(0), _u)
    condition_on!(x, Proj(1), _du)
    condition_on!(x, Proj(2), _f(_du, _u))

    set_variables("u", numvars=2d, order=q+1)

    fp1 = taylor_expand(_f, stacked_u)
    fp2 = taylor_expand(u -> u[1:d], stacked_u)
    f_derivatives = [fp1]
    for o in 3:q
        _curr_f_deriv = f_derivatives[end]
        dfdu1 = stack([derivative.(_curr_f_deriv, i) for i in 1:d])'
        dfdu2 = stack([derivative.(_curr_f_deriv, i) for i in d+1:2d])'
        # dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        # df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        df = dfdu1 * fp1 + dfdu2 * fp2
        push!(f_derivatives, df)
        condition_on!(x, Proj(o), evaluate(df))
    end

    return nothing
end
