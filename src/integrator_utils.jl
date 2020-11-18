# Calibration, smoothing, then jump to the OrdinaryDiffEq._postamble!
function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEKF})

    if isstatic(integ.cache.diffusionmodel) # Calibrate
        # @warn "sol.log_likelihood is not correct for static diffusion models!"
        integ.sol.log_likelihood = NaN
        final_diff = integ.sol.diffusions[end]
        diff_sqrt = final_diff .^ (1/2)
        for s in integ.sol.x
            # s.Σ .*= final_diff
            copy!(s.Σ, PSDMatrix(LowerTriangular(diff_sqrt .* s.Σ.L)))
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
        @assert (length(integ.sol.u) == length(integ.sol.pu))
        [(su .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
    end


    OrdinaryDiffEq._postamble!(integ)
end


function DiffEqBase.savevalues!(
    integrator::OrdinaryDiffEq.ODEIntegrator{<:AbstractEKF},
    force_save=false, reduce_size=true)

    @assert integrator.opts.dense
    @assert integrator.saveiter == integrator.saveiter_dense

    # Do whatever OrdinaryDiffEq would do
    out = OrdinaryDiffEq._savevalues!(integrator, force_save, reduce_size)

    # stuff that would normally be in apply_step!
    copy!(integrator.cache.x, integrator.cache.x_filt)

    # Save our custom stuff that we need for the posterior
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.x, integrator.saveiter, copy(integrator.cache.x))
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.diffusions, integrator.saveiter, copy(integrator.cache.diffmat))
    OrdinaryDiffEq.copyat_or_push!(integrator.sol.pu, integrator.saveiter, integrator.cache.E0*integrator.cache.x)

    integrator.sol.log_likelihood += integrator.cache.log_likelihood

    return out
end
