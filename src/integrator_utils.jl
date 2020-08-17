"""accept/reject previous step and chose new dt
"This is done at the top so that way the interator interface happens mid-step instead of post-step"
"""
function loopheader!(integ)
end


"""Calculate new timesteps, save, apply the callbacks"""
function loopfooter!(integ)
    integ.accept_step, dt_proposal = integ.steprule(integ)

    # proposal = (t=integ.t_new,
    #             prediction=integ.cache.x_pred,
    #             filter_estimate=integ.cache.x_filt,
    #             measurement=integ.cache.measurement,
    #             H=integ.H, Q=integ.Qh, v=h,
    #             σ²=σ_sq)
    # push!(integ.proposals, (integ.proposal..., accept=integ.accept_step, dt=integ.dt))

    integ.dt = dt_proposal

    if integ.accept_step
        integ.cache.x = copy(integ.cache.x_filt)
        integ.u = integ.constants.E0 * integ.cache.x.μ
        integ.t = integ.t_new

        integ.cache.σ_sq_prev = integ.cache.σ_sq

        # For the solution
        push!(integ.state_estimates, integ.cache.x)
        push!(integ.times, integ.t)

        integ.destats.naccept += 1
    else
        integ.destats.nreject += 1
    end

    # TODO: Add check for maxiters back in again
end


"""This could handle smoothing and uncertainty-calibration"""
function postamble!(integ)
    integ.smooth && smooth!(integ)
    calibrate!(integ)
end
