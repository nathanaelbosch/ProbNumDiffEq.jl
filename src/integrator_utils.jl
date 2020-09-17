"""accept/reject previous step and chose new dt

"This is done at the top so that way the interator interface happens mid-step instead of post-step"
UPDATE 17-09-2020:
Had trouble with correclty handling the exact step to T, so I moved the "apply step" stuff
into loopfooter!.
"""
function loopheader!(integ)
    integ.iter += 1
    fix_dt_at_bounds!(integ)
end

function fix_dt_at_bounds!(integ)
    integ.dt = min(integ.opts.dtmax, integ.dt)
    integ.dt = max(integ.opts.dtmin, integ.dt)
    next_t = integ.t + integ.dt
    if next_t + integ.opts.dtmin > integ.prob.tspan[2]
        # Avoid having to make a step smaller than dtmin in the next step
        integ.dt = integ.prob.tspan[2] - integ.t
    end
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
    isnan(integ.cache.σ_sq) && error("Estimated sigma is NaN")
    integ.opts.adaptive && isnan(integ.EEst) && error("Error estimate is NaN")
    isnan(integ.dt) && error("Step size is NaN")



    # Accept or reject the step: Moved this here from loopheader!
    if integ.iter > 0 && integ.accept_step
        integ.success_iter += 1
        apply_step!(integ)
    end
end


"""This could handle smoothing and uncertainty-calibration"""
function postamble!(integ)
    if isstatic(integ.sigma_estimator)
        calibrate!(integ)
        integ.sigmas .= integ.sigmas[end]
    end
    integ.smooth && smooth_all!(integ)
end


function apply_step!(integ)
    copy!(integ.cache.x, integ.cache.x_filt)
    mul!(integ.u, integ.constants.E0, integ.cache.x.μ)
    integ.t = integ.t_new

    integ.cache.σ_sq_prev = integ.cache.σ_sq

    # For the solution
    push!(integ.state_estimates, copy(integ.cache.x))
    push!(integ.times, copy(integ.t))
    push!(integ.sigmas, copy(integ.cache.σ_sq))
end
