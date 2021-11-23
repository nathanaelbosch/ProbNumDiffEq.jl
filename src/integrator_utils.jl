" ProbNumDiffEq.jl-specific implementation of OrdinaryDiffEq.jl's `postamble!`."
function OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK})
    # OrdinaryDiffEq.jl-related calls:
    OrdinaryDiffEq._postamble!(integ)
    copyat_or_push!(integ.sol.k, integ.saveiter_dense, integ.k)
    pn_solution_endpoint_match_cur_integrator!(integ)

    # Calibrate the solution (if applicable)
    if isstatic(integ.cache.diffusionmodel)
        if integ.cache.diffusionmodel.calibrate
            # The estimated global_diffusion is just a scaling factor
            mle_diffusion = integ.cache.global_diffusion
            calibrate_solution!(integ, mle_diffusion)
        else
            constant_diffusion = integ.cache.default_diffusion
            set_diffusions!(integ.sol, constant_diffusion)
        end
    end

    if integ.alg.smooth
        smooth_solution!(integ)
    end

    @assert (length(integ.sol.u) == length(integ.sol.pu))

    return nothing
end

"""
    calibrate_solution!(integ, mle_diffusion)

Calibrate the solution (`integ.sol`) with the specified `mle_diffusion` by (i) setting the
values in `integ.sol.diffusions` to the `mle_diffusion` (see [`set_diffusions!`](@ref)),
(ii) rescaling all filtering estimates such that they have the correct diffusion, and (iii)
updating the solution estimates in `integ.sol.pu`.
"""
function calibrate_solution!(integ, mle_diffusion)

    # Set all solution diffusions; don't forget the initial diffusion!
    set_diffusions!(integ.sol, mle_diffusion * integ.cache.default_diffusion)

    # Rescale all filtering estimates to have the correct diffusion
    @simd ivdep for s in integ.sol.x_filt
        copy!(s.Σ, apply_diffusion(s.Σ, mle_diffusion))
    end

    # Re-write into the solution estimates
    for (pu, x) in zip(integ.sol.pu, integ.sol.x_filt)
        mul!(pu, integ.cache.SolProj, x)
    end
    # [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
end

"""
    set_diffusions!(solution::AbstractProbODESolution, diffusion)

Set the contents of `solution.diffusions` to the provided `diffusion`, overwriting the local
diffusion estimates that are in there. Typically, `diffusion` is either a global quasi-MLE
or the specified initial diffusion value if no calibration is desired.
"""
function set_diffusions!(solution::AbstractProbODESolution, diffusion::Number)
    solution.diffusions .= diffusion
    return nothing
end
function set_diffusions!(solution::AbstractProbODESolution, diffusion::Diagonal)
    @simd ivdep for d in solution.diffusions
        copy!(d, diffusion)
    end
    return nothing
end

"""
    smooth_solution!(integ)

Smoothes the solution saved in `integ.sol`, filling `integ.sol.x_smooth` and updating the
values saved in `integ.sol.pu` and `integ.sol.u`. This function handles the iteration and
preconditioning. The actual smoothing step happens in [`smooth!`](@ref).
"""
function smooth_solution!(integ)
    integ.sol.x_smooth = copy(integ.sol.x_filt)

    @unpack A, Q = integ.cache
    @unpack x_smooth, t, diffusions = integ.sol
    @unpack x_tmp, x_tmp2 = integ.cache
    x = x_smooth

    for i in length(x)-1:-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x[i], x[i+1])
            continue
        end

        make_preconditioners!(integ.cache, dt)
        P, PI = integ.cache.P, integ.cache.PI

        mul!(x_tmp, P, x[i])
        mul!(x_tmp2, P, x[i+1])
        smooth!(x_tmp, x_tmp2, A, Q, integ, diffusions[i])
        mul!(x[i], PI, x_tmp)

        # Save the smoothed state into the solution
        mul!(integ.sol.pu[i], integ.cache.SolProj, x[i])
        integ.sol.u[i][:] .= integ.sol.pu[i].μ
    end
    integ.sol.interp = set_smooth(integ.sol.interp)
    return nothing
end

"Inspired by `OrdinaryDiffEq.solution_match_cur_integrator!`"
function pn_solution_endpoint_match_cur_integrator!(integ)
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

"Extends `OrdinaryDiffEq._savevalues!` to save ProbNumDiffEq.jl-specific things."
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
