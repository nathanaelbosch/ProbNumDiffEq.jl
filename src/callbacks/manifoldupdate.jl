function manifoldupdate!(cache, residualf; maxiters=10, steptol=nothing)
    x_pred = cache.x

    # Skip update if cov is exactly zero
    iszero(x_pred.Σ.R) && return nothing

    @unpack SolProj, tmp, x_tmp, x_tmp2 = cache
    z_tmp = residualf(mul!(tmp, SolProj, x_pred.μ))
    result = DiffResults.JacobianResult(z_tmp, tmp)
    d = length(z_tmp)
    d <= cache.d || throw(
        DimensionMismatch(
            "The residual function returned a $d-dimensional residual, but the ODE is " *
            "only $(cache.d)-dimensional; `ManifoldUpdate` requires " *
            "`length(residual(u)) <= length(u)`."))

    H = view(cache.H, 1:d, :)
    obs_cache = make_obssized_cache(cache; o=d)

    steptol = isnothing(steptol) ? sqrt(eps(eltype(x_pred.μ))) : steptol

    # Linearize at the current iterate m_i, and update the prediction
    x_out = x_tmp
    m_i = x_pred.μ
    for i in 1:maxiters
        u_i = mul!(tmp, SolProj, m_i)
        ForwardDiff.jacobian!(result, residualf, u_i)
        mul!(H, DiffResults.jacobian(result), SolProj)
        obs = LinearizedObservation(m_i, DiffResults.value(result), H)

        (; S, K, B) = try
            update_mean!(x_out, x_pred, obs; cache=obs_cache)
        catch e
            e isa PosDefException ? manifold_rankerror(u_i) : rethrow()
        end
        length(S) == 1 && iszero(S[1]) && manifold_rankerror(u_i)

        if norm(x_out.μ .- m_i) <= steptol * norm(x_out.μ) || i == maxiters
            update_cov!(x_out, x_pred, obs, K, B; cache=obs_cache)
            break
        end
        m_i = copy!(x_tmp2.μ, x_out.μ)
    end

    copy!(cache.x, x_out)
    return nothing
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
    ManifoldUpdate(residual::Function; maxiters=10, steptol=sqrt(eps(T)))

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
- `steptol`: The iteration stops once a step changes the mean of the state by at most
  `steptol` relative to it. The iteration converges quadratically, so the default, the
  square root of the machine epsilon of the state's element type `T`, leaves an error at
  the level of the machine epsilon.
"""
function ManifoldUpdate(
    residual::Function,
    args...;
    maxiters=10,
    steptol=nothing,
    kwargs...,
)
    condition(u, t, integ) = true
    affect!(integ) = begin
        manifoldupdate!(integ.cache, residual; maxiters, steptol)
        mul!(view(integ.u, :), integ.cache.SolProj, integ.cache.x.μ)
    end
    return DiscreteCallback(condition, affect!, args...; kwargs...)
end
