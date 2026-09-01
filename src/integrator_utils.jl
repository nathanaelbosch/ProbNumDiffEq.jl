"""
    SciMLBase.postamble!(integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK})

ProbNumDiffEq.jl-specific implementation of SciMLBase's `postamble!` hook.

In addition to calling `OrdinaryDiffEqCore._postamble!(integ)`, calibrate the diffusion and
smooth the solution.
"""
function SciMLBase.postamble!(
    integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK},
)
    # OrdinaryDiffEqCore.jl-related calls:
    OrdinaryDiffEqCore._postamble!(integ)
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

    @assert (length(integ.sol.u) == length(integ.sol.pu) == length(integ.sol.x_filt))

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
    @assert mle_diffusion isa Diagonal
    @simd ivdep for C in integ.sol.x_filt.Σ
        apply_diffusion!(C, mle_diffusion)
    end
    @simd ivdep for C in integ.sol.backward_kernels.C
        apply_diffusion!(C, mle_diffusion)
    end

    # Re-write into the solution estimates
    for i in eachindex(integ.sol.pu, integ.sol.x_filt)
        _gaussian_mul!(integ.sol.pu[i], integ.cache.SolProj, integ.sol.x_filt[i])
    end
    # [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
end

"""
    set_diffusions!(solution::AbstractProbODESolution, diffusion::Union{Number,Diagonal})

Set the contents of `solution.diffusions` to the provided `diffusion`, overwriting the local
diffusion estimates that are in there. Typically, `diffusion` is either a global quasi-MLE
or the specified initial diffusion value if no calibration is desired.
"""
function set_diffusions!(solution::AbstractProbODESolution, diffusion)
    if diffusion isa Diagonal{<:Number,<:FillArrays.Fill}
        @simd ivdep for i in eachindex(solution.diffusions)
            solution.diffusions[i] = copy(diffusion)
        end
    elseif diffusion isa Diagonal{<:Number,<:Vector}
        @simd ivdep for d in solution.diffusions
            copy!(d, diffusion)
        end
    else
        throw(ArgumentError("unexpected diffusion type $(typeof(diffusion))"))
    end
    return nothing
end

"""
    smooth_solution!(integ)

Smooth the solution saved in `integ.sol`, filling `integ.sol.x_smooth` and updating the
values saved in `integ.sol.pu` and `integ.sol.u`.

This function handles the iteration and preconditioning.
The actual smoothing step happens by [`marginalize!`](@ref)ing backward kernels.
"""
function smooth_solution!(integ)
    @unpack cache, sol = integ
    append!(sol.x_smooth, sol.x_filt)

    @unpack x_smooth, t, backward_kernels = sol
    @unpack C_DxD, C_3DxD = cache

    @assert length(x_smooth) == length(backward_kernels) + 1

    for i in (length(x_smooth)-1):-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x_smooth[i], x_smooth[i+1])
            continue
        end

        K = backward_kernels[i]
        marginalize!(x_smooth[i], x_smooth[i+1], K; C_DxD, C_3DxD)

        _gaussian_mul!(sol.pu[i], cache.SolProj, x_smooth[i])
        sol.u[i][:] .= sol.pu[i].μ
    end
    return nothing
end

"Inspired by `OrdinaryDiffEqCore.solution_match_cur_integrator!`"
function pn_solution_endpoint_match_cur_integrator!(integ)
    if integ.opts.save_end
        i = integ.saveiter

        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)

        copyat_or_push!(
            integ.sol.pu,
            i,
            _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x),
        )

        if !integ.opts.save_everystep && i > 1
            save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
        end
    end
end

"""
    save_diffusion!(sol, i, diffusion)

`copyat_or_push!` for `sol.diffusions`, which needs to replace entries instead of copying
into them: the scalar diffusions are `Fill`-backed `Diagonal`s and cannot be mutated.
"""
function save_diffusion!(sol, i, diffusion)
    if i <= length(sol.diffusions)
        sol.diffusions[i] = copy(diffusion)
    else
        push!(sol.diffusions, copy(diffusion))
    end
    return nothing
end

"Extends `OrdinaryDiffEqCore._savevalues!` to save ProbNumDiffEq.jl-specific things."
function DiffEqBase.savevalues!(
    integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK},
    force_save=false,
    reduce_size=true,
)

    # Do whatever OrdinaryDiffEqCore would do
    out = OrdinaryDiffEqCore._savevalues!(integ, force_save, reduce_size)

    # Save our custom stuff that we need for the posterior
    if integ.opts.save_everystep
        i = integ.saveiter
        save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)
        _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x)
        copyat_or_push!(integ.sol.pu, i, integ.cache.pu_tmp)

        if integ.alg.smooth
            copyat_or_push!(
                integ.sol.backward_kernels, i, integ.cache.backward_kernel)
        end
    end

    return out
end

function OrdinaryDiffEqCore.update_uprev!(
    integ::OrdinaryDiffEqCore.ODEIntegrator{<:AbstractEK},
)
    @assert !OrdinaryDiffEqCore.alg_extrapolates(integ.alg)
    @assert isinplace(integ.sol.prob)
    @assert !(integ.alg isa OrdinaryDiffEqCore.DAEAlgorithm)

    recursivecopy!(integ.uprev, integ.u)
    recursivecopy!(integ.cache.xprev, integ.cache.x)
    nothing
end
