########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################
function smooth_all!(integ)

    @unpack x, t, diffusions = integ.sol
    @unpack A, Q, Precond = integ.cache
    # x_pred is just used as a cache here

    for i in length(x)-1:-1:2
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x[i], x[i+1])
            continue
        end

        P = Precond(dt)
        PI = inv(P)

        Qh = apply_diffusion(Q, diffusions[i])

        x[i] = P * x[i]
        smooth!(x[i], P*x[i+1], A, Qh, integ)
        any(isnan.(x[i].μ)) && error("NaN mean after smoothing")
        any(isnan.(x[i].Σ)) && error("NaN cov after smoothing")
        x[i] = PI * x[i]
    end
end


function smooth!(x_curr, x_next, Ah, Qh, integ)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    @unpack d, q = integ.cache
    @unpack x_tmp = integ.cache

    # @info "smooth!" x_curr.Σ x_next.Σ Ah Qh PI
    if all((Qh) .< eps(eltype(Qh)))
        @warn "smooth: Qh is really small! The system is basically deterministic, so we just \"predict backwards\"."
        return inv(Ah) * x_next
    end

    # Prediction: t -> t+1
    predict!(x_tmp, x_curr, Ah, Qh)

    # Smoothing
    P_p = x_tmp.Σ
    P_p_inv = inv(P_p)
    G = x_curr.Σ * Ah' * P_p_inv
    x_curr.μ .+= G * (x_next.μ .- x_tmp.μ)

    # Joseph-Form:
    K_tilde = x_curr.Σ * Ah' * P_p_inv
    P_s = (
        X_A_Xt(x_curr.Σ, (I - K_tilde*Ah))
        + X_A_Xt(Qh, K_tilde)
        + X_A_Xt(x_next.Σ, G)
    )
    copy!(x_curr.Σ, P_s)

    assert_nonnegative_diagonal(x_curr.Σ)
    # PDMat(Symmetric(x_curr.Σ))

    return nothing
end
