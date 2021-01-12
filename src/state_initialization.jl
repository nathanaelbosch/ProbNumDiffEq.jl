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
        dfdu = stack([derivative.(_curr_f_deriv, i) for i in 1:d])'
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






# DAE STUFF
function iip_to_oop(f!::DAEFunction)
    function f(du, u, p, t)
        out = copy(u)
        f!(out, du, u, p, t)
        return out
    end
    return f
end
function initialize_without_derivatives(u0, du0, f::DAEFunction, p, t0, order, var=1e-3)
    q = order
    d = length(u0)

    m0 = zeros(d*(q+1))
    m0[1:d] = u0

    f = isinplace(f) ? iip_to_oop(f) : f
    @assert iszero(f(du0, u0, p, t0))

    d = length(u0)
    q = order
    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

    m0[1:d] .= u0
    m0[d+1:2d] .= du0

    P0 = Matrix([zeros(2d, 2d) zeros(2d, d*(q-1));
          zeros(d*(q-1), 2d) Diagonal(var .* ones(d*(q-1)))])

    return m0, P0
end
