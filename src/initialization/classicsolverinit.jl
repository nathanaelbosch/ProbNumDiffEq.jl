function initial_update!(integ, cache, ::ClassicSolverInit)
    @unpack u, f, p, t = integ
    @unpack d, q, x, Proj = cache
    @assert isinplace(f)

    if q > 5
        @warn "ClassicSolverInit might be unstable for high orders"
    end

    @unpack ddu, du, x_tmp, m_tmp, K1 = cache
    @unpack x_tmp, K1, C_Dxd, C_DxD, C_dxd, measurement = cache

    # Initialize on u0; taking special care for DynamicalODEProblems
    is_secondorder = integ.f isa DynamicalODEFunction
    _u = is_secondorder ? view(u.x[2], :) : view(u, :)
    E0 = x.Σ.R isa IsoKroneckerProduct ? Proj(0) : Matrix(Proj(0))
    init_condition_on!(x, E0, _u, cache)
    is_secondorder ? f.f1(du, u.x[1], u.x[2], p, t) : f(du, u, p, t)
    integ.stats.nf += 1
    E1 = x.Σ.R isa IsoKroneckerProduct ? Proj(1) : Matrix(Proj(1))
    init_condition_on!(x, E1, view(du, :), cache)

    if q < 2
        return
    end

    # Use a jac or autodiff to initialize on ddu0
    if f isa ODEFunction && integ.alg.initialization.init_on_ddu
        _f = if f.f isa SciMLBase.FunctionWrappersWrappers.FunctionWrappersWrapper
            ODEFunction(SciMLBase.unwrapped_f(f), mass_matrix=f.mass_matrix)
        else
            f
        end

        dfdt = copy(u)
        ForwardDiff.derivative!(dfdt, (du, t) -> _f(du, u, p, t), du, t)

        if !isnothing(f.jac)
            f.jac(ddu, u, p, t)
        else
            ForwardDiff.jacobian!(ddu, (du, u) -> _f(du, u, p, t), du, u)
        end
        ddfddu = ddu * view(du, :) + view(dfdt, :)
        E2 = x.Σ.R isa IsoKroneckerProduct ? Proj(2) : Matrix(Proj(2))
        init_condition_on!(x, E2, ddfddu, cache)
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
    integ.stats.nf += 2

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
    integ.stats.nf += sol.stats.nf
    integ.stats.njacs += sol.stats.njacs
    integ.stats.nsolve += sol.stats.nsolve
    integ.stats.nw += sol.stats.nw
    integ.stats.nnonliniter += sol.stats.nnonliniter
    integ.stats.nnonlinconvfail += sol.stats.nnonlinconvfail
    integ.stats.ncondition += sol.stats.ncondition

    # Filter & smooth to fit these values!
    us = [u for u in sol.u]
    return rk_init_improve(cache, sol.t, us, dt)
end

function rk_init_improve(cache::AbstractODEFilterCache, ts, us, dt)
    @unpack A, Q = cache
    @unpack x, x_pred, x_filt, measurement = cache
    @unpack K1, C_Dxd, C_DxD, C_dxd = cache

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
        predict!(x_pred, x, A, Q, cache.C_DxD, cache.C_2DxD, cache.default_diffusion)
        push!(preds, copy(x_pred))

        H = cache.E0 * PI
        measurement.μ .= H * x_pred.μ .- u
        fast_X_A_Xt!(measurement.Σ, x_pred.Σ, H)

        update!(x_filt, x_pred, measurement, H, K1, C_Dxd, C_DxD, C_dxd)
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
