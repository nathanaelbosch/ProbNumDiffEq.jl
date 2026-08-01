function manifoldupdate!(cache, residualf; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15)
    m, C = mean(cache.x), cov(cache.x)

    @unpack SolProj, tmp, x_tmp = cache
    D = cache.d * (cache.q + 1)
    z_tmp = residualf(mul!(tmp, SolProj, m))
    result = DiffResults.JacobianResult(z_tmp, tmp)
    d = length(z_tmp)
    if d > cache.d
        throw(
            DimensionMismatch(
                "The residual function returned a $d-dimensional residual, but the ODE " *
                "is only $(cache.d)-dimensional. `ManifoldUpdate` requires " *
                "`length(residual(u)) <= length(u)`."),
        )
    end

    _H = view(cache.H, 1:d, :)
    _K1 = view(cache.C_2DxD, 1:D, 1:d)
    _K2 = view(cache.C_2DxD, (D+1):(2D), 1:d)
    M_cache = cache.C_DxD
    S = PSDMatrix(view(cache.C_Dxd, :, 1:d))
    S_gram = view(cache.C_dxd, 1:d, 1:d)

    m_tmp, C_tmp = mean(x_tmp), cov(x_tmp)

    m_i = copy(m)
    for i in 1:maxiters
        u_i = mul!(tmp, SolProj, m_i)

        ForwardDiff.jacobian!(result, residualf, u_i)
        z = DiffResults.value(result)
        J = DiffResults.jacobian(result)

        mul!(_H, J, SolProj)
        fast_X_A_Xt!(S, C, _H)  # S.R = C.R * H'

        # m_i_new, C_i_new = update(x, Gaussian(z .+ (H * (m - m_i)), S), H)
        _S = make_hermitian_if_fowarddiff(_matmul!(S_gram, S.R', S.R))
        S_chol = if length(_S) == 1
            iszero(_S[1]) && rankerror(u_i)
            _S[1]
        else
            cholesky_or_rankerror!(_S, u_i)
        end
        copyto!(_K1, S.R)
        rdiv!(_K1, S_chol)
        K = _matmul!(_K2, C.R', _K1)

        m_tmp .= m_i .- m
        mul!(z_tmp, _H, m_tmp)
        z_tmp .-= z
        mul!(m_tmp, K, z_tmp)
        m_i_new = m_tmp .+= m

        if (norm(m_i_new .- m_i) < ϵ₁ && norm(z) < ϵ₂) || (i == maxiters)
            # C_i_new = X_A_Xt(C, I - K * H)
            _matmul!(M_cache, K, _H, -1.0, 0.0)
            @inbounds @simd ivdep for j in 1:D
                M_cache[j, j] += 1
            end
            fast_X_A_Xt!(C_tmp, C, M_cache)
            break
        end
        m_i = m_i_new
    end

    copy!(cache.x, Gaussian(m_tmp, C_tmp))

    return nothing
end

rankerror(u) = throw(
    ArgumentError(
        "The measurement covariance of the `ManifoldUpdate` is singular at u = $u. " *
        "This means that the Jacobian of the provided residual function does not have " *
        "full row rank there, e.g. because the residual contains redundant or " *
        "identically-zero components. Provide a residual function with only as many " *
        "components as there are independent constraints."),
)

function cholesky_or_rankerror!(S, u)
    try
        return cholesky!(S)
    catch e
        e isa LinearAlgebra.PosDefException || rethrow()
        rankerror(u)
    end
end

"""
    ManifoldUpdate(residual::Function)

Update the state to satisfy a zero residual function via iterated extended Kalman filtering.

`ManifoldUpdate` returns a `SciMLBase.DiscreteCallback`, which, at each solver step,
performs an iterated extended Kalman filter update to keep the residual measurement to be
zero. Additional arguments and keyword arguments for the `DiscreteCallback` can be passed.

The residual function should be `residual(u::AbstractVector)::AbstractVector`, that is
_it should not be in-place_ (whereas DiffEqCallback.jl's `ManifoldProjection`) is.

The residual should have exactly one component per independent constraint, and in
particular `length(residual(u)) <= length(u)`; it does _not_ need to have the same shape as
`u`. Its Jacobian must have full row rank, so do not pad the residual with
identically-zero components to match `length(u)`: that makes the measurement covariance
singular and raises an error.

# Additional keyword arguments
- `maxiters::Int`: Maximum number of IEKF iterations.
  Setting this to 1 results in a single standard EKF update.
"""
function ManifoldUpdate(
    residual::Function,
    args...;
    maxiters=100,
    ϵ₁=1e-25,
    ϵ₂=1e-15,
    kwargs...,
)
    condition(u, t, integ) = true
    affect!(integ) = begin
        manifoldupdate!(integ.cache, residual; maxiters=maxiters, ϵ₁=ϵ₁, ϵ₂=ϵ₂)
        mul!(view(integ.u, :), integ.cache.SolProj, integ.cache.x.μ)
    end
    return DiscreteCallback(condition, affect!, args...; kwargs...)
end
