"""accept/reject previous step and chose new dt

"This is done at the top so that way the interator interface happens mid-step instead of post-step"
"""
function loopheader!(integ)

    # Accept or reject the step
    if integ.iter > 0 && integ.accept_step
        apply_step!(integ)
    end

    integ.dt = min(integ.opts.dtmax, integ.dt)
    integ.dt = max(integ.opts.dtmin, integ.dt)

    integ.iter += 1
end


"""Calculate new timesteps, save, apply the callbacks"""
function loopfooter!(integ)

    integ.accept_step = integ.opts.adaptive ? integ.EEst < 1 : true


    push!(integ.proposals, (
        t=integ.t_new,
        prediction=copy(integ.cache.x_pred),
        filter_estimate=copy(integ.cache.x_filt),
        measurement=copy(integ.cache.measurement),
        H=copy(integ.cache.H), Q=copy(integ.cache.Qh), v=copy(integ.cache.h),
        σ²=copy(integ.cache.σ_sq),
        accept=integ.accept_step,
        dt=integ.dt
    ))

    integ.opts.adaptive && (integ.dt = propose_step(integ.steprule, integ))

    integ.accept_step ? (integ.destats.naccept += 1) : (integ.destats.nreject += 1)

    # TODO: Add check for maxiters back in again
end


"""This could handle smoothing and uncertainty-calibration"""
function postamble!(integ)
    if isstatic(integ.sigma_estimator)
        calibrate!(integ)
        integ.sigmas .= integ.sigmas[end]
    end
    integ.smooth && smooth!(integ)
end


function apply_step!(integ)
    copy!(integ.cache.x, integ.cache.x_filt)
    mul!(integ.u, integ.constants.E0, integ.cache.x.μ)
    integ.t = integ.t_new

    integ.cache.σ_sq_prev = integ.cache.σ_sq

    # For the solution
    push!(integ.state_estimates, copy(integ.cache.x))
    push!(integ.times, integ.t)
    push!(integ.sigmas, integ.cache.σ_sq)
end
