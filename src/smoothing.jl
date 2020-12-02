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
    @unpack x_tmp, x_tmp2, G, covmatcache = integ.cache
    veccache = x_tmp.μ

    # Prediction: t -> t+1
    predict!(x_tmp, x_curr, Ah, Qh, G)

    # Smoothing
    P_p = x_tmp.Σ
    P_p_inv = inv(P_p)

    # Compute G withouth additional allocations
    mul!(covmatcache, Ah', P_p_inv)
    mul!(G, x_curr.Σ, covmatcache)

    mul!(veccache, G, (x_next.μ - x_tmp.μ))
    x_curr.μ .+= veccache

    # Joseph-Form:
    mul!(covmatcache, G, Ah)
    P_s = (
        X_A_Xt(x_curr.Σ, (I - covmatcache))
        + X_A_Xt(Qh, G)
        + X_A_Xt(x_next.Σ, G)
    )
    copy!(x_curr.Σ, P_s)

    # assert_nonnegative_diagonal(x_curr.Σ)
    # PDMat(Symmetric(x_curr.Σ))

    return nothing
end
