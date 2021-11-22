# Calibration, smoothing, then jump to the OrdinaryDiffEq._postamble!
function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK})
    OrdinaryDiffEq._postamble!(integ)
    # For some unknown reason, the following is necessary
    copyat_or_push!(integ.sol.k, integ.saveiter_dense, integ.k)

    # Add the final timepoint to the solution
    pn_solution_endpoint_match_cur_integrator!(integ)

    if isstatic(integ.cache.diffusionmodel)
        if integ.cache.diffusionmodel isa FixedDiffusion &&
           !integ.cache.diffusionmodel.calibrate
            # Set all diffusions to the final diffusion
            final_diff = integ.cache.diffusionmodel.initial_diffusion
            set_diffusions(integ, final_diff)
        else
            integ.sol.log_likelihood = NaN
            final_diff =
                integ.cache.global_diffusion * integ.cache.diffusionmodel.initial_diffusion

            set_diffusions(integ, final_diff)

            # Rescale all filtering estimates to have the correct diffusion
            rescale_diff = final_diff ./ integ.cache.diffusionmodel.initial_diffusion
            @simd ivdep for s in integ.sol.x_filt
                copy!(s.Σ, apply_diffusion(s.Σ, rescale_diff))
            end

            # Re-write into the solution estimates
            for (pu, x) in zip(integ.sol.pu, integ.sol.x_filt)
                mul!(pu, integ.cache.SolProj, x)
            end
            [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
        end
    end

    if integ.alg.smooth
        smooth_all!(integ)
        for (pu, x) in zip(integ.sol.pu, integ.sol.x_smooth)
            mul!(pu, integ.cache.SolProj, x)
        end
        integ.sol.interp = set_smooth(integ.sol.interp)
        [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
    end
    @assert (length(integ.sol.u) == length(integ.sol.pu))

    return nothing
end
function set_diffusions(integ, final_diff)
    if isempty(size(final_diff))
        integ.sol.diffusions .= final_diff
    else
        [(d .= final_diff) for d in integ.sol.diffusions]
    end
end

function pn_solution_endpoint_match_cur_integrator!(integ)
    # Inspired from OrdinaryDiffEq.solution_match_cur_integrator!
    if integ.opts.save_end
        if integ.alg.smooth
            OrdinaryDiffEq.copyat_or_push!(
                integ.sol.x_filt,
                integ.saveiter_dense,
                integ.cache.x,
            )
        end

        OrdinaryDiffEq.copyat_or_push!(
            integ.sol.pu,
            integ.saveiter,
            mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x),
        )
    end
end

function DiffEqBase.savevalues!(
    integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK},
    force_save=false,
    reduce_size=true,
)

    # Do whatever OrdinaryDiffEq would do
    out = OrdinaryDiffEq._savevalues!(integ, force_save, reduce_size)

    # Save our custom stuff that we need for the posterior
    # TODO If we don't want dense output, we might not want to save these!
    # It's not completely clear how to specify that though; They are also needed for sampling.
    if integ.alg.smooth
        OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, integ.cache.x)
    end
    OrdinaryDiffEq.copyat_or_push!(
        integ.sol.diffusions,
        integ.saveiter,
        integ.cache.local_diffusion,
    )
    if integ.opts.save_everystep
        OrdinaryDiffEq.copyat_or_push!(
            integ.sol.pu,
            integ.saveiter,
            mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x),
        )
    end

    return out
end
