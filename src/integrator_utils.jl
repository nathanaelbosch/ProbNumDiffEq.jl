# Calibration, smoothing, then jump to the OrdinaryDiffEq._postamble!
function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEKF})

    if isstatic(integ.cache.diffusionmodel) # Calibrate
        @warn "sol.log_likelihood is not correct for static diffusion models!"
        final_diff = integ.sol.diffusions[end]
        for s in integ.sol.x
            s.Σ .*= final_diff
        end

        if isempty(size(final_diff))
            integ.sol.diffusions .= final_diff
        else
            [(d .= final_diff) for d in integ.sol.diffusions]
        end
    end

    if integ.alg.smooth
        smooth_all!(integ)
        integ.sol.pu .= [integ.cache.E0 * x for x in integ.sol.x]
        integ.sol.u .= integ.sol.pu.μ
    end


    OrdinaryDiffEq._postamble!(integ)
end


function OrdinaryDiffEq.savevalues!(
    integrator::OrdinaryDiffEq.ODEIntegrator{<:AbstractEKF},
    force_save=false, reduce_size=true)

    @assert integrator.opts.dense
    @assert integrator.saveiter == integrator.saveiter_dense

    # Do whatever OrdinaryDiffEq would do
    OrdinaryDiffEq._savevalues!(integrator, force_save, reduce_size)

    # stuff that would normally be in apply_step!
    copy!(integrator.cache.x, integrator.cache.x_filt)

    # Save our custom stuff that we need for the posterior
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.x, integrator.saveiter, copy(integrator.cache.x))
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.diffusions, integrator.saveiter, copy(integrator.cache.diffmat))
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.pu, integrator.saveiter, integrator.cache.E0*integrator.cache.x)

    integrator.sol.log_likelihood += integrator.cache.log_likelihood
end
