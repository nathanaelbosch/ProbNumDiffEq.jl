function initial_update!(integ, cache, init::RungeKuttaInit)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = cache
    q = integ.alg.order

    @unpack du, x_tmp, x_tmp2, m_tmp, K1, K2 = cache

    t0 = integ.sol.prob.tspan[1]
    dt = 0.01
    nsteps = q + 1
    tmax = t0 + nsteps*dt
    tstops = t0:dt:tmax
    alg = integ.alg isa EK0 ? Vern9()  : Rodas5()
    sol = solve(remake(integ.sol.prob, tspan=(t0, tmax)),
                alg, adaptive=false, dense=false, tstops=tstops, save_start=false)

    # Initialize on u0
    condition_on!(x, Proj(0), u, m_tmp, K1, K2, x_tmp.Σ, x_tmp2.Σ.mat)

    # Initialize on du0
    if isinplace(f)
        f(du, u, p, t)
    else
        du .= f(u, p, t)
    end
    condition_on!(x, Proj(1), du, m_tmp, K1, K2, x_tmp.Σ, x_tmp2.Σ.mat)

    if q > 1
        # Filter & smooth to fit these values!
        rk_init_improve(integ, cache, sol.t, sol.u, dt)
    end

end

function rk_init_improve(integ, cache::GaussianODEFilterCache, ts, us, dt)
    @unpack A, Q, Ah, Qh = cache
    @unpack x, x_pred, x_filt, measurement = cache

    # Predict forward:
    make_preconditioners!(integ, dt)
    @unpack P, PI = cache

    mul!(x, P, x)

    preds = []
    filts = [copy(x)]

    # Filter through the data forwards
    for (i, (t, u)) in enumerate(zip(ts, us))

        predict_mean!(x_pred, x, A)
        predict_cov!(x_pred, x, A, Q)
        push!(preds, copy(x_pred))

        H = cache.E0 * PI
        measurement.μ .= H * x_pred.μ .- u
        X_A_Xt!(measurement.Σ, x_pred.Σ, H)

        update!(x_filt, x_pred, measurement, H, 0,
                cache.K1, cache.K2, cache.x_tmp2.Σ.mat)
        push!(filts, copy(x_filt))

        x = x_filt

    end

    # Smooth backwards
    for i in length(filts):-1:2
        xf = filts[i-1]
        xs = filts[i]
        xp = preds[i-1] # Since `preds` is one shorter
        smooth!(xf, xs, A, Q, integ, 1)
    end

    mul!(cache.x, PI, filts[1])

    return nothing
end
