function initial_update!(integ, cache, init::TaylorModeInit)
    @unpack u, f, p, t = integ
    @unpack d, q, x, Proj = cache
    q = integ.alg.order
    D = d * (q + 1)

    @unpack x_tmp, K1 = cache
    if size(K1, 2) != d
        K1 = K1[:, 1:d]
    end

    if f isa ODEFunction &&
       f.f isa SciMLBase.FunctionWrappersWrappers.FunctionWrappersWrapper
        f = ODEFunction(SciMLBase.unwrapped_f(f), mass_matrix=f.mass_matrix)
    end

    f_derivatives = taylormode_get_derivatives(u, f, p, t, q)
    integ.destats.nf += q
    @assert length(0:q) == length(f_derivatives)
    m_cache = Gaussian(zeros(eltype(u), d), PSDMatrix(zeros(eltype(u), D, d)))
    for (o, df) in zip(0:q, f_derivatives)
        if f isa DynamicalODEFunction
            @assert df isa ArrayPartition
            df = df[2, :]
        end
        pmat = f.mass_matrix * Proj(o)

        if !(df isa AbstractVector)
            df = df[:]
        end

        condition_on!(x, pmat, df, cache)
    end
end

"""
    Compute initial derivatives of an IIP ODEProblem with TaylorIntegration.jl
"""
function taylormode_get_derivatives(u, f::AbstractODEFunction{true}, p, t, q)
    tT = Taylor1(typeof(t), q)
    tT[0] = t
    uT = similar(u, Taylor1{eltype(u)})
    @inbounds @simd ivdep for i in eachindex(u)
        uT[i] = Taylor1(u[i], q)
    end
    @info "error?" uT
    duT = zero(uT)
    uauxT = similar(uT)
    TaylorIntegration.jetcoeffs!(f, tT, uT, duT, uauxT, p)
    return [evaluate.(differentiate.(uT, i)) for i in 0:q]
end
