"""initialize x0 up to the provided order"""
function initial_update!(integ)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = integ.cache
    q = integ.alg.order

    f_derivatives = get_derivatives(u, f, p, t, q)
    @assert length(0:q) == length(f_derivatives)
    for (o, df) in zip(0:q, f_derivatives)
        condition_on!(x, Proj(o), df)
    end
end


"""
    Compute initial derivatives of an ODEProblem with TaylorSeries.jl
"""
function get_derivatives(u, f, p, t, q)
    d = length(u)

    f_oop = isinplace(f) ? iip_to_oop(f) : f

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(_t -> f_oop(u, p, _t), t)
    @assert !(eltype(f_t_taylor) <: TaylorN) "The vector field depends on t; The code might not yet be able to handle these (but it should be easy to implement)"

    # Simplify further:
    _f(u) = f_oop(u, p, t)

    u0 = u
    du0 = _f(u)

    set_variables("u", numvars=d, order=q+1)

    fp = taylor_expand(_f, u)
    f_derivatives = [fp]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu = stack([TaylorSeries.derivative.(_curr_f_deriv, i) for i in 1:d])'
        df = dfdu * fp
        push!(f_derivatives, df)
    end

    return [u, evaluate.(f_derivatives)...]
end


"""
    Compute initial derivatives of a SecondOrderODE with TaylorSeries.jl
"""
function get_derivatives(u::ArrayPartition, f::DynamicalODEFunction, p, t, q)

    d = length(u[1,:])
    Proj(deriv) = deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    f_oop(du, u, p, t) = (ddu = copy(du); f.f1(ddu, du, u, p, t); return ddu)

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
