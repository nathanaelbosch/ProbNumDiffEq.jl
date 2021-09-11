# Calibration, smoothing, then jump to the OrdinaryDiffEq._postamble!
function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK})
    if isstatic(integ.cache.diffusionmodel) # Calibrate
        # @warn "sol.log_likelihood is not correct for static diffusion models!"
        integ.sol.log_likelihood = NaN
        final_diff = integ.sol.diffusions[end]
        @simd ivdep for s in integ.sol.x_filt
            # s.Σ .*= final_diff
            copy!(s.Σ, apply_diffusion(s.Σ, final_diff))
        end

        if isempty(size(final_diff))
            integ.sol.diffusions .= final_diff
        else
            [(d .= final_diff) for d in integ.sol.diffusions]
        end
    end

    if integ.alg.smooth
        smooth_all!(integ)
        for (pu, x) in zip(integ.sol.pu, integ.sol.x_smooth)
            mul!(pu, integ.cache.SolProj, x)
        end
        integ.sol.interp = set_smooth(integ.sol.interp)
        @assert (length(integ.sol.u) == length(integ.sol.pu))
        [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
    end

    return OrdinaryDiffEq._postamble!(integ)
end

function DiffEqBase.savevalues!(
    integrator::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK},
    force_save=false,
    reduce_size=true,
)

    # Do whatever OrdinaryDiffEq would do
    out = OrdinaryDiffEq._savevalues!(integrator, force_save, reduce_size)

    # Save our custom stuff that we need for the posterior
    # TODO If we don't want dense output, we might not want to save these!
    # It's not completely clear how to specify that though; They are also needed for sampling.
    if integrator.alg.smooth
        OrdinaryDiffEq.copyat_or_push!(
            integrator.sol.x_filt,
            integrator.saveiter,
            integrator.cache.x,
        )
    end
    OrdinaryDiffEq.copyat_or_push!(
        integrator.sol.diffusions,
        integrator.saveiter,
        integrator.cache.global_diffusion,
    )
    if integrator.opts.save_everystep
        OrdinaryDiffEq.copyat_or_push!(
            integrator.sol.pu,
            integrator.saveiter,
            mul!(integrator.cache.pu_tmp, integrator.cache.SolProj, integrator.cache.x),
        )
    end

    return out
end
