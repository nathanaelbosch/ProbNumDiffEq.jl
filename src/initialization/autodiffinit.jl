function initial_update!(integ, cache, init::AutodiffInitializationScheme)
    @unpack u, f, p, t = integ
    @unpack d, q, q, x, Proj = cache
    D = d * (q + 1)

    @unpack x_tmp, K1, C_Dxd, C_DxD, C_dxd, measurement = cache
    if size(K1, 2) != d
        K1 = K1[:, 1:d]
    end

    if f isa ODEFunction &&
       f.f isa SciMLBase.FunctionWrappersWrappers.FunctionWrappersWrapper
        f = ODEFunction(SciMLBase.unwrapped_f(f), mass_matrix=f.mass_matrix)
    end

    f_derivatives = get_derivatives(init, u, f, p, t)
    integ.stats.nf += init.order
    @assert length(f_derivatives) == init.order+1

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
            df = df[2, :]
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
function get_derivatives(init::TaylorModeInit, u, f::SciMLBase.AbstractODEFunction{true}, p, t)
    q = init.order
    tT = Taylor1(typeof(t), q)
    tT[0] = t
    uT = similar(u, Taylor1{eltype(u)})
    @inbounds @simd ivdep for i in eachindex(u)
        uT[i] = Taylor1(u[i], q)
    end
    duT = zero(uT)
    uauxT = similar(uT)
    TaylorIntegration.jetcoeffs!(f, tT, uT, duT, uauxT, p)
    # return hcat([evaluate.(differentiate.(uT, i)) for i in 0:q]...)'
    return [evaluate.(differentiate.(uT, i)) for i in 0:q]
end

function get_derivatives(init::ForwardDiffInit, u, f::SciMLBase.AbstractODEFunction{true}, p, t)
    q = init.order
    _f(u) = (du = copy(u); f(du, u, p, t); du)
    f_n = _f
    out = [u]
    push!(out, _f(u))
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


function forwarddiff_get_derivatives!(out, u, f::SciMLBase.AbstractODEFunction{true}, p, t, q)
    _f(du, u) = f(du, u, p, t)
    d = length(u0)
    f_n = _f
    out[1:d] .= u0
    @views _f(out[d+1:2d], u0)
    for o in 2:ndiffs
        f_n = forwarddiff_iip_vectorfield_derivative_iteration(f_n, _f)
        @views f_n(out[o*d+1:(o+1)*d], u0)
    end
    return out
end

function forwarddiff_iip_vectorfield_derivative_iteration(f_n, f_0)
    function df(du, u)
        J = ForwardDiff.jacobian(f_n, du, u)
        f_0(du, u)
        _matmul!(du, J, du)
    end
    return df
end
