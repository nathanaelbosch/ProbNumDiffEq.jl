function manifoldupdate!(cache, residualf; maxiters=100, ϵ₁=1e-25, ϵ₂=1e-15)
    m, C = mean(cache.x), cov(cache.x)

    # If the state covariance is exactly zero (e.g. right after `TaylorModeInit`, before
    # the covariance has grown), the innovation covariance `H C Hᵀ` is zero too, and the
    # Kalman gain is undefined. With zero covariance the filter has no uncertainty to
    # correct, so skip the update instead of computing a degenerate gain.
    iszero(C.R) && return nothing

    @unpack SolProj, tmp, x_tmp, x_tmp2 = cache
    D = cache.d * (cache.q + 1)
    z_tmp = residualf(mul!(tmp, SolProj, m))
    result = DiffResults.JacobianResult(z_tmp, tmp)
    d = length(z_tmp)
    d <= cache.d || throw(
        DimensionMismatch(
            "The residual function returned a $d-dimensional residual, but the ODE is " *
            "only $(cache.d)-dimensional; `ManifoldUpdate` requires " *
            "`length(residual(u)) <= length(u)`."))

    _H = view(cache.H, 1:d, :)
    _K1 = view(cache.C_2DxD, 1:D, 1:d)
    _K2 = view(cache.C_2DxD, (D+1):(2D), 1:d)
    S = PSDMatrix(view(cache.C_Dxd, :, 1:d))
    S_gram = view(cache.C_dxd, 1:d, 1:d)

    m_tmp, C_tmp = mean(x_tmp), cov(x_tmp)

    m_i = copy!(mean(x_tmp2), m)
    for i in 1:maxiters
        u_i = mul!(tmp, SolProj, m_i)

        ForwardDiff.jacobian!(result, residualf, u_i)
        z = DiffResults.value(result)
        J = DiffResults.jacobian(result)

        mul!(_H, J, SolProj)
        fast_X_A_Xt!(S, C, _H)  # S.R = C.R * H'

        # m_i_new, C_i_new = update(x, Gaussian(z .+ (H * (m - m_i)), S), H)
        S_chol = cholesky_or_rankerror!(
            make_hermitian_if_fowarddiff(_matmul!(S_gram, S.R', S.R)), u_i)
        copyto!(_K1, S.R)
        rdiv!(_K1, S_chol)
        K = _matmul!(_K2, C.R', _K1)

        m_tmp .= m_i .- m
        mul!(z_tmp, _H, m_tmp)
        z_tmp .-= z
        mul!(m_tmp, K, z_tmp)
        m_i_new = m_tmp .+= m

        if (norm(z) < ϵ₂ && norm(m_i_new .- m_i) < ϵ₁) || (i == maxiters)
            # C_tmp.R = C.R * (I - K * H)' = C.R - S.R * K'
            copy!(C_tmp.R, C.R)
            _matmul!(C_tmp.R, S.R, K', -1.0, 1.0)
            break
        end
        m_i = m_i_new
    end

    copy!(cache.x, Gaussian(m_tmp, C_tmp))

    return nothing
end

function cholesky_or_rankerror!(S, u)
    if length(S) == 1
        iszero(S[1]) && manifold_rankerror(u)
        return S[1]
    end
    chol = cholesky!(Symmetric(S), check=false)
    issuccess(chol) || manifold_rankerror(u)
    return chol
end

manifold_rankerror(u) = throw(
    ArgumentError(
        "The measurement covariance of the `ManifoldUpdate` is singular at u = $u. " *
        "Usually this means that the Jacobian of the residual function does not have " *
        "full row rank there, e.g. because the residual contains redundant or " *
        "identically-zero components; it can also happen if the state covariance " *
        "itself has become singular."),
)

"""
    ManifoldUpdate(residual::Function)

Update the state to satisfy a zero residual function via iterated extended Kalman filtering.

`ManifoldUpdate` returns a `SciMLBase.DiscreteCallback`, which, at each solver step,
performs an iterated extended Kalman filter update to keep the residual measurement to be
zero. Additional arguments and keyword arguments for the `DiscreteCallback` can be passed.

The residual function should be `residual(u::AbstractVector)::AbstractVector`, that is
_it should not be in-place_ (whereas DiffEqCallback.jl's `ManifoldProjection`) is.

The residual should have one component per independent constraint, so
`length(residual(u)) <= length(u)`; it does _not_ need to have the same shape as `u`.
Its Jacobian must have full row rank.

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
