function initial_update!(integ, cache, ::ClassicSolverInit)
    @unpack u, f, p, t = integ
    @unpack d, q, x, Proj = cache
    @assert isinplace(f)

    if q > 5
        @warn "ClassicSolverInit might be unstable for high orders"
    end

    @unpack ddu, du = cache

    # Initialize on u0; taking special care for DynamicalODEProblems
    is_secondorder = integ.f isa DynamicalODEFunction
    _u = is_secondorder ? view(u.x[2], :) : view(u, :)
    init_condition_on!(x, Proj(0), _u, cache)
    is_secondorder ? u.x[1] : f(du, u, p, t)
    integ.stats.nf += 1
    init_condition_on!(x, Proj(1), view(du, :), cache)

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
        init_condition_on!(x, Proj(2), ddfddu, cache)
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
    # @unpack Ah, Qh = cache
    @unpack x, x_pred, x_filt, measurement = cache
    @unpack K1, C_Dxd, C_DxD, C_dxd, C_3DxD, C_d = cache
    @unpack backward_kernel = cache

    # Predict forward:
    make_preconditioners!(cache, dt)
    @unpack P, PI = cache

    _gaussian_mul!(x, P, x)

    preds = []
    filts = [copy(x)]
    backward_kernels = []

    # Filter through the data forwards
    for (i, (t, u)) in enumerate(zip(ts, us))
        (u isa RecursiveArrayTools.ArrayPartition) && (u = u.x[2]) # for 2ndOrderODEs
        u = view(u, :) # just in case the problem is matrix-valued

        predict!(x_pred, x, A, Q, cache.C_DxD, cache.C_2DxD, cache.default_diffusion)
        push!(preds, copy(x_pred))

        K = AffineNormalKernel(A, Q)
        compute_backward_kernel!(
            backward_kernel, x_pred, x, K; C_DxD, diffusion=cache.default_diffusion)
        push!(backward_kernels, copy(backward_kernel))

        H = cache.E0 * PI
        measurement.μ .= H * x_pred.μ .- u
        fast_X_A_Xt!(measurement.Σ, x_pred.Σ, H)

        update!(x_filt, x_pred, measurement, H, K1, C_Dxd, C_DxD, C_dxd, C_d)
        push!(filts, copy(x_filt))

        x = x_filt
    end

    # Smooth backwards
    x_smooth = filts
    for i in length(x_smooth)-1:-1:1
        marginalize!(x_smooth[i], x_smooth[i+1], backward_kernels[i]; C_DxD, C_3DxD)
    end

    _gaussian_mul!(cache.x, PI, filts[1])

    return nothing
end
