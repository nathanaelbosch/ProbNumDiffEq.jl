"""initialize x0 up to the provided order"""
function initial_update!(integ, cache, init::TaylorModeInit)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = cache
    q = integ.alg.order

    @unpack x_tmp, x_tmp2, m_tmp, K1, K2 = cache

    f_derivatives = taylormode_get_derivatives(u, f, p, t, q)
    @assert length(0:q) == length(f_derivatives)
    for (o, df) in zip(0:q, f_derivatives)
        if f isa DynamicalODEFunction
            @assert df isa ArrayPartition
            df = df[2, :]
        end
        pmat = f.mass_matrix * Proj(o)
        condition_on!(x, pmat, view(df, :), m_tmp, K1, x_tmp.Σ, x_tmp2.Σ.mat)
    end
end

"""
    Compute initial derivatives of an OOP ODEProblem with TaylorSeries.jl
"""
function taylormode_get_derivatives(u, f::AbstractODEFunction{false}, p, t, q)
    f = oop_to_iip(f)

    tT = Taylor1(typeof(t), q)
    tT[0] = t
    uT = Taylor1.(u, tT.order)
    duT = zero.(Taylor1.(u, tT.order))
    uauxT = similar(uT)
    TaylorIntegration.jetcoeffs!(f, tT, uT, duT, uauxT, p)
    return [evaluate.(differentiate.(uT, i)) for i in 0:q]
end
"""
    Compute initial derivatives of an IIP ODEProblem with TaylorIntegration.jl
"""
function taylormode_get_derivatives(u, f::AbstractODEFunction{true}, p, t, q)
    tT = Taylor1(typeof(t), q)
    tT[0] = t
    uT = Taylor1.(u, tT.order)
    duT = zero.(Taylor1.(u, tT.order))
    uauxT = similar(uT)
    TaylorIntegration.jetcoeffs!(f, tT, uT, duT, uauxT, p)
    return [evaluate.(differentiate.(uT, i)) for i in 0:q]
end
