"""accept/reject previous step and chose new dt
"This is done at the top so that way the interator interface happens mid-step instead of post-step"
"""
function loopheader!(integ)
end


"""Calculate new timesteps, save, apply the callbacks"""
function loopfooter!(integ)
    integ.accept_step, dt_proposal = integ.steprule(integ)
    push!(integ.proposals, (integ.proposal..., accept=integ.accept_step, dt=integ.dt))
    integ.dt = dt_proposal

    if integ.accept_step 
        integ.x = integ.proposal.filter_estimate
        integ.t = integ.proposal.t
        push!(integ.state_estimates, (t=integ.t, x=integ.x))
    end

    # TODO: Add check for maxiters back in again
end


"""This could handle smoothing and uncertainty-calibration"""
function postamble!(integ)
    integ.smooth && smooth!(integ.state_estimates, integ)
    calibrate!(integ.state_estimates, integ)
end