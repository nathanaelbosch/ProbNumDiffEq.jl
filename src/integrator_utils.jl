"""
    OrdinaryDiffEq.postamble!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK})

ProbNumDiffEq.jl-specific implementation of OrdinaryDiffEq.jl's `postamble!`.

In addition to calling `OrdinaryDiffEq._postamble!(integ)`, calibrate the diffusion and
smooth the solution.
"""
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
    @simd ivdep for S in integ.sol.x_filt.Σ
        if mle_diffusion isa Number
            S.R .*= sqrt(mle_diffusion)
        elseif mle_diffusion isa Diagonal
            S.R .= S.R .* sqrt.(mle_diffusion.diag)'
        else
            error()
        end
    end
    @simd ivdep for S in integ.sol.backward_kernels.C
        if mle_diffusion isa Number
            S.R .*= sqrt(mle_diffusion)
        elseif mle_diffusion isa Diagonal
            S.R .= S.R .* sqrt.(mle_diffusion.diag)'
        else
            error()
        end
    end

    # Re-write into the solution estimates
    for (pu, x) in zip(integ.sol.pu, integ.sol.x_filt)
        _gaussian_mul!(pu, integ.cache.SolProj, x)
    end
    # [(su[:] .= pu) for (su, pu) in zip(integ.sol.u, integ.sol.pu.μ)]
end

"""
    set_diffusions!(solution::AbstractProbODESolution, diffusion::Union{Number,Diagonal})

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

Smooth the solution saved in `integ.sol`, filling `integ.sol.x_smooth` and updating the
values saved in `integ.sol.pu` and `integ.sol.u`.

This function handles the iteration and preconditioning.
The actual smoothing step happens in [`smooth!`](@ref).
"""
function smooth_solution!(integ)
    integ.sol.x_smooth = copy(integ.sol.x_filt)

    @unpack x_smooth, t, backward_kernels = integ.sol
    @unpack C_DxD, C_2DxD = integ.cache

    @assert length(x_smooth) == length(backward_kernels) + 1

    for i in length(x_smooth)-1:-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x_smooth[i], x_smooth[i+1])
            continue
        end

        K = backward_kernels[i]

        # marginalize!(x_smooth[i], x_smooth[i+1], K; C_DxD, C_2DxD)

        make_transition_matrices!(integ.cache, dt)
        @unpack P, PI = integ.cache
        @unpack x_tmp, x_tmp2, backward_kernel = integ.cache
        K_tmp = backward_kernel
        _gaussian_mul!(x_tmp, P, x_smooth[i])
        _gaussian_mul!(x_tmp2, P, x_smooth[i+1])
        _matmul!(K_tmp.A, P, _matmul!(C_DxD, K.A, PI))
        _matmul!(K_tmp.b, P, K.b)
        X_A_Xt!(K_tmp.C, K.C, PI)
        marginalize!(x_tmp, x_tmp2, K_tmp; C_DxD, C_2DxD)
        _gaussian_mul!(x_smooth[i], PI, x_tmp)

        # Save the smoothed state into the solution
        _gaussian_mul!(integ.sol.pu[i], integ.cache.SolProj, x_smooth[i])
        integ.sol.u[i][:] .= integ.sol.pu[i].μ
    end
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
            _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x),
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
    if integ.opts.save_everystep
        i = integ.saveiter
        OrdinaryDiffEq.copyat_or_push!(integ.sol.diffusions, i, integ.cache.local_diffusion)
        OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)
        _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x)
        OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, i, integ.cache.pu_tmp)

        if integ.alg.smooth
            OrdinaryDiffEq.copyat_or_push!(
                integ.sol.backward_kernels, i, integ.cache.backward_kernel)
        end
    end

    return out
end

function OrdinaryDiffEq.update_uprev!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK})
    @assert !OrdinaryDiffEq.alg_extrapolates(integ.alg)
    @assert isinplace(integ.sol.prob)
    @assert !(integ.alg isa OrdinaryDiffEq.DAEAlgorithm)

    recursivecopy!(integ.uprev, integ.u)
    recursivecopy!(integ.cache.xprev, integ.cache.x)
    nothing
end
