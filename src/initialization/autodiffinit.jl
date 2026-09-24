function initial_update!(integ, cache, init::AutodiffInitializationScheme)
    @unpack u, f, p, t = integ
    @unpack d, q, x, Proj = cache

    if f isa ODEFunction &&
       f.f isa SciMLBase.FunctionWrappersWrappers.FunctionWrappersWrapper
        f = ODEFunction(SciMLBase.unwrapped_f(f), mass_matrix=f.mass_matrix)
    end

    f_derivatives = get_derivatives(init, u, f, p, t)
    integ.stats.nf += init.order
    @assert length(f_derivatives) == init.order + 1

    # This is hacky and should definitely be removed. But it also works so 🤷
    MM = if f.mass_matrix isa UniformScaling
        f.mass_matrix
    else
        _MM = copy(f.mass_matrix)
        if any(iszero.(diag(_MM)))
            _MM = typeof(promote(_MM[1], 1e-20)[1]).(_MM)
            _MM .+= 1e-20I(d)
        end
        _MM
    end

    for (o, df) in zip(0:q, f_derivatives)
        if f isa DynamicalODEFunction
            @assert df isa ArrayPartition
            df = df.x[2]
        end

        df = view(df, :)

        H = if o == 0
            Proj(o)
        else
            MM * Proj(o)
        end
        init_condition_on!(x, H, df, cache)
    end
end

"""
    Compute initial derivatives of an IIP ODEProblem with TaylorIntegration.jl
"""
function get_derivatives(
    init::TaylorModeInit, u, f::SciMLBase.AbstractODEFunction{true}, p, t)
    f_as_Function(du, u, p, t) = f(du, u, p, t)
    q = init.order
    tT = Taylor1(typeof(t), q)
    tT[0] = t
    uT = similar(u, Taylor1{eltype(u)})
    @inbounds @simd ivdep for i in eachindex(u)
        uT[i] = Taylor1(u[i], q)
    end
    duT = zero(uT)
    uauxT = similar(uT)
    TaylorIntegration.jetcoeffs!(f_as_Function, tT, uT, duT, uauxT, p)
    # return hcat([evaluate.(differentiate.(uT, i)) for i in 0:q]...)'
    return [evaluate.(differentiate.(uT, i)) for i in 0:q]
end

function get_derivatives(
    init::ForwardDiffInit, u, f::SciMLBase.AbstractODEFunction{true}, p, t)
    q = init.order
    _f(u) = (du=copy(u); f(du, u, p, t); du)

    out = [u]
    push!(out, _f(u))

    f_n = _f
    for _ in 2:q
        f_n = forwarddiff_oop_vectorfield_derivative_iteration(f_n, _f)
        push!(out, f_n(u))
    end

    return out
end

function forwarddiff_oop_vectorfield_derivative_iteration(f_n, f_0)
    function df(u)
        J = ForwardDiff.jacobian(f_n, u)
        return J * f_0(u)
    end
    return df
end
