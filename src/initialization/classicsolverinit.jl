function initial_update!(integ, cache, ::ClassicSolverInit)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = cache
    q = integ.alg.order
    @assert isinplace(f)

    if q > 5
        @warn "ClassicSolverInit might be unstable for high orders"
    end

    @unpack ddu, du, x_tmp, x_tmp2, m_tmp, K1 = cache

    # Initialize on u0; taking special care for DynamicalODEProblems
    is_secondorder = integ.f isa DynamicalODEFunction
    _u = is_secondorder ? view(u.x[2], :) : view(u, :)
    Mcache = cache.C_DxD
    condition_on!(x, Proj(0), _u, m_tmp.Σ, K1, x_tmp.Σ, Mcache, cache)
    is_secondorder ? f.f1(du, u.x[1], u.x[2], p, t) : f(du, u, p, t)
    integ.destats.nf += 1
    condition_on!(x, Proj(1), view(du, :), m_tmp.Σ, K1, x_tmp.Σ, Mcache, cache)

    if q < 2
        return
    end

    # Use a jac or autodiff to initialize on ddu0
    if integ.alg.initialization.init_on_du
        dfdt = copy(u)
        ForwardDiff.derivative!(dfdt, (du, t) -> f(du, u, p, t), du, t)

        if !isnothing(f.jac)
            f.jac(ddu, u, p, t)
        else
            ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u)
        end
        ddfddu = ddu * view(du, :) + view(dfdt, :)
        condition_on!(x, Proj(2), ddfddu, m_tmp.Σ, K1, x_tmp.Σ, Mcache, cache)
        if q < 3
            return
        end
    end

    # Compute the other parts with classic solvers
    t0 = integ.sol.prob.tspan[1]
    dt =
        10 * OrdinaryDiffEq.ode_determine_initdt(
            u,
            t,
            1,
            integ.opts.dtmax,
            integ.opts.abstol,
            integ.opts.reltol,
            integ.opts.internalnorm,
            integ.sol.prob,
            integ,
        )
    integ.destats.nf += 2

    nsteps = q + 2
    tmax = t0 + nsteps * dt
    tstops = t0:dt:tmax
    alg = integ.alg.initialization.alg
    sol = solve(
        remake(integ.sol.prob, tspan=(t0, tmax)),
        alg,
        dense=false,
        save_start=false,
        abstol=integ.opts.abstol / 100,
        reltol=integ.opts.reltol / 100,
        saveat=tstops,
    )
    # This is necessary in order to fairly account for the cost of initialization!
    integ.destats.nf += sol.destats.nf
    integ.destats.njacs += sol.destats.njacs
    integ.destats.nsolve += sol.destats.nsolve
    integ.destats.nw += sol.destats.nw
    integ.destats.nnonliniter += sol.destats.nnonliniter
    integ.destats.nnonlinconvfail += sol.destats.nnonlinconvfail
    integ.destats.ncondition += sol.destats.ncondition

    # Filter & smooth to fit these values!
    us = [u for u in sol.u]
    return rk_init_improve(cache, sol.t, us, dt)
end

function rk_init_improve(cache::GaussianODEFilterCache, ts, us, dt)
    @unpack A, Q = cache
    @unpack x, x_pred, x_filt, measurement = cache

    # Predict forward:
    make_preconditioners!(cache, dt)
    @unpack P, PI = cache

    _gaussian_mul!(x, P, x)

    preds = []
    filts = [copy(x)]

    # Filter through the data forwards
    for (i, (t, u)) in enumerate(zip(ts, us))
        (u isa RecursiveArrayTools.ArrayPartition) && (u = u.x[2]) # for 2ndOrderODEs
        u = view(u, :) # just in case the problem is matrix-valued
        predict_mean!(x_pred, x, A)
        predict_cov!(x_pred, x, A, Q, cache.C_DxD, cache.C_2DxD, cache.default_diffusion)
        push!(preds, copy(x_pred))

        H = cache.E0 * PI
        measurement.μ .= H * x_pred.μ .- u
        X_A_Xt!(measurement.Σ, x_pred.Σ, H)

        update!(x_filt, x_pred, measurement, H, cache.K1, cache.C_DxD, cache.m_tmp.Σ)
        push!(filts, copy(x_filt))

        x = x_filt
    end

    # Smooth backwards
    for i in length(filts):-1:2
        xf = filts[i-1]
        xs = filts[i]
        xp = preds[i-1] # Since `preds` is one shorter
        smooth!(xf, xs, A, Q, cache, 1)
    end

    _gaussian_mul!(cache.x, PI, filts[1])

    return nothing
end
