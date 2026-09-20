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

!!! warning "Call at most once per solution"
    Step (ii) rescales the stored quantities **in place** -- the filtering covariances
    `integ.sol.x_filt.Σ`, the backward-kernel covariances `integ.sol.backward_kernels.C`,
    and the measurement Cholesky factors `SmootherState.S_U` in `integ.sol.smoother_states`
    (see [`SmootherState`](@ref)) are all multiplied by the new diffusion, rather than
    recomputed from an unscaled reference. The operation is therefore not idempotent: a
    second call (or a manual diffusion rescale followed by a call) would apply the scaling
    twice and silently corrupt the solution's uncertainties and the pre-stored smoother
    states. There is exactly one call site, the `SciMLBase.postamble!` hook, which
    runs once per solve and before [`smooth_solution!`](@ref); keep it that way.
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

    # Keep the pre-stored smoother-state measurement quantities consistent with the
    # recalibrated covariances: the Kalman gains `K = Σ_pred Hᵀ S⁻¹` are invariant under
    # the rescaling (the measurement Jacobians of EK0 and DiagonalEK1 are dimension-pure,
    # and calibration with observation noise is ruled out by `ekargcheck`), while the
    # measurement covariances `S = H Σ_pred Hᵀ` pick up the same per-dimension congruence
    # as `Σ_pred` - so their Cholesky factors scale by the matching square root.
    mle_scale = _measurement_chol_scale(mle_diffusion)
    for ss in integ.sol.smoother_states
        ss === nothing && continue
        _rescale_measurement_chol!(ss.S_U, mle_scale)
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

# Project the smoothed state at index `i` into solution space and write it into the
# solution's `pu`/`u` arrays.
function _store_smoothed!(sol, cache, i)
    _gaussian_mul!(sol.pu[i], cache.SolProj, sol.x_smooth[i])
    sol.u[i][:] .= sol.pu[i].μ
    return nothing
end

"""
    smooth_solution!(integ)

Smooth the solution saved in `integ.sol`, filling `integ.sol.x_smooth` and updating the
values saved in `integ.sol.pu` and `integ.sol.u`.

Two implementations are available, selected by the solver's `smoother` argument:

- `:mbf` (default): the square-root Modified Bryson--Frazier (√MBF) smoother (Gibbs 2011),
  which propagates adjoint variables backward and only inverts the `d×d` measurement
  covariance (not the full `D×D` predicted state covariance as in the RTS smoother). The
  λ/Λ update and predict recursion is done in sqrt form in each step's local preconditioned
  coordinates for numerical stability; covariance recovery uses a hyperbolic QR
  factorization to guarantee a positive semi-definite result. The per-step measurement
  quantities are pre-stored by the forward pass in [`SmootherState`](@ref).
- `:rts`: the classic Rauch--Tung--Striebel smoother, marginalizing the backward transition
  kernels that were computed (in preconditioned square-root form) by the forward pass.

The actual per-step work is done by [`_mbf_backward_step!`](@ref) (MBF) or
`marginalize!` (RTS), both dispatching on the structure of the state covariances (dense,
EK0's Kronecker, or DiagonalEK1's block-diagonal), the same way [`predict_cov!`](@ref) and
[`update!`](@ref) do -- so the loops don't need to know or care which algorithm produced
the solution being smoothed.
"""
function smooth_solution!(integ)
    @unpack cache, sol = integ
    for (i, x) in enumerate(sol.x_filt)
        copyat_or_push!(sol.x_smooth, i, x)
    end

    if integ.alg.smoother == :mbf
        _smooth_solution_mbf!(integ)
    else
        _smooth_solution_rts!(integ)
    end
    return nothing
end

"MBF branch of [`smooth_solution!`](@ref); see there."
function _smooth_solution_mbf!(integ)
    @unpack cache, sol = integ
    @unpack x_smooth, t, smoother_states = sol

    # One smoother state per transition t[j] -> t[j+1]; the backward loop indexes
    # `smoother_states[i-1]` directly, so a bookkeeping desync would otherwise surface as a
    # raw `BoundsError`. Mirrors the equivalent guard in `_smooth_solution_rts!`.
    @assert length(smoother_states) == length(x_smooth) - 1 "smoother_states must hold one entry per transition (t[j] → t[j+1]); got $(length(smoother_states)) for $(length(x_smooth)) smoothed states"

    n = length(x_smooth)

    λ = zeros(eltype(x_smooth[1].μ), length(x_smooth[1].μ))
    # U_Λ needs to hold a general (non-diagonal) matrix once measurement info accumulates,
    # so it must be zero-initialized from the state covariance's structure (dense/Kronecker/
    # block-diagonal), not from `cache.P`/`PI` -- those are diagonal preconditioners even in
    # the dense (EK1) case, and a `Diagonal`-typed `U_Λ` can't hold the general update result.
    U_Λ = zero(x_smooth[1].Σ.R)

    # Preallocate scratch buffers once and reuse them across all backward steps.
    scratch = _mbf_scratch(U_Λ, cache)

    # x_smooth[n] = x_filt[n] exactly: there are no future measurements to smooth with.
    _store_smoothed!(sol, cache, n)

    # The backward loop restores per-step parameters into the cache's prior (see
    # `_step_rates`); snapshot the forward-final value so the cache can be left at it
    # afterwards rather than at the earliest step's (the loop's last restoration), matching
    # what a non-smoothed solve leaves behind, in case anything downstream inspects
    # `cache.prior` post-solve.
    saved_rate_parameter = _step_rates(cache.prior)

    for i in n:-1:2
        if iszero(t[i] - t[i-1])
            copy!(x_smooth[i-1], x_smooth[i])
            _store_smoothed!(sol, cache, i - 1)
            continue
        end

        ss = smoother_states[i-1]
        dt_step = t[i] - t[i-1]
        # Restore the per-step parameters of this step's transition (see
        # `_restore_step_rates!`): `cache.prior` still holds the value of the last forward
        # step, but the transition matrices of this step were computed with this step's
        # value.
        if ss isa SmootherState
            _restore_step_rates!(cache.prior, ss.rate_parameter)
        end
        make_transition_matrices!(cache, cache.prior, dt_step)

        λ, U_Λ = _mbf_backward_step!(
            x_smooth[i-1], λ, U_Λ, sol.x_filt[i-1], ss,
            cache.P, cache.PI, cache.A, scratch)

        _store_smoothed!(sol, cache, i - 1)
    end

    # Leave the cache's prior at the forward-final value (see above).
    _restore_step_rates!(cache.prior, saved_rate_parameter)

    return nothing
end

"RTS branch of [`smooth_solution!`](@ref); see there."
function _smooth_solution_rts!(integ)
    @unpack cache, sol = integ
    @unpack x_smooth, t, backward_kernels = sol
    @unpack C_DxD, C_3DxD = cache

    @assert length(x_smooth) == length(backward_kernels) + 1

    for i in (length(x_smooth)-1):-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x_smooth[i], x_smooth[i+1])
        else
            marginalize!(x_smooth[i], x_smooth[i+1], backward_kernels[i]; C_DxD, C_3DxD)
        end
        _store_smoothed!(sol, cache, i)
    end
    return nothing
end

"""
    _save_smoother_state!(smoother_states, cache)

Store the measurement quantities that the √MBF backward smoother needs for this step's
update: the measurement mean `z = h(x_pred)` (left in `cache.measurement.μ`; the filter's
innovation is `0 - z`, since the update assumes zero measurements), the measurement Jacobian
`H`, the Kalman gain `K` (left in `cache.C_Dxd` by `update!`), and the upper triangular
Cholesky factor `S_U` of the measurement covariance (written to `cache.measurement_chol` by
`update!`; it includes `R` if observation noise is configured).

`K` and `S_U` are exactly the quantities the forward filter's update used, so backward
smoothing with them reproduces the filter's posterior exactly; see
[`calibrate_solution!`](@ref) for how they stay consistent when the solution is
calibrated after the solve.

Called once per saved step and appends one entry, which keeps the array aligned with
`sol.t`: `smoother_states[j]` holds the transition `sol.t[j] -> sol.t[j+1]`, so
`length(smoother_states) == length(sol.t) - 1`. The caller only runs this when
`OrdinaryDiffEqCore._savevalues!` actually saved; runs without a save (e.g. the final
step under `save_end=false`) belong to no stored transition and are skipped there.
"""
function _save_smoother_state!(smoother_states, cache)
    if iszero(cache.x_pred.Σ.R) || iszero(cache.measurement.Σ)
        # Degenerate step: `update!` skipped the update (zero predicted covariance,
        # mirroring `update!`'s own early-exit condition), or the measurement covariance
        # is exactly zero (e.g. `Σ_pred ≠ 0` but `H Σ_pred H' = 0` with no observation
        # noise). Either way there is no measurement information to smooth with, so the
        # backward pass at this step does a plain prediction (`ss === nothing` branch of
        # `_mbf_backward_step!`).
        push!(smoother_states, nothing)
        return nothing
    end
    T = eltype(cache.C_Dxd)
    H = copy(cache.H)
    z = Vector{T}(cache.measurement.μ)
    K = copy(cache.C_Dxd)
    S_U = _extract_measurement_chol(cache.measurement_chol)
    # Snapshot the prior's per-step parameters (see `_step_rates`): for the IOUP prior
    # with `update_rate_parameter=true`, `calc_J!` mutates the array on the cache's prior
    # at the start of every step, so the backward pass needs a copy of this step's value.
    rate = _step_rates(cache.prior)
    push!(smoother_states, SmootherState(z, H, K, S_U, rate))
    return nothing
end

"Inspired by `OrdinaryDiffEqCore.solution_match_cur_integrator!`"
function pn_solution_endpoint_match_cur_integrator!(integ)
    if integ.opts.save_end
        i = integ.saveiter

        # `OrdinaryDiffEqCore.solution_endpoint_match_cur_integrator!` (called just before
        # this, from `_postamble!`) may have appended a save point that the step's own
        # `savevalues!` did not save. This happens when solving onto an existing time grid
        # (`solve(prob, alg, u, t, k)`, used e.g. by DiffEqDevTools' timing loops): the
        # `t`/`u` arrays are prefilled, so `_savevalues!`'s `integ.t !== sol.t[end]` check
        # suppresses the save of the final step whose time already is the prefilled
        # `sol.t[end]`. The per-step quantities of that step then still have to be
        # appended here, or they end up out of sync with `sol.t`/`x_filt`.
        appended_new_step = i > length(integ.sol.x_filt)

        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)

        copyat_or_push!(
            integ.sol.pu,
            i,
            _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x),
        )

        if integ.opts.save_everystep && appended_new_step
            _save_step_quantities!(integ, i)
        elseif !integ.opts.save_everystep && i > 1
            save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
        end
    end
end

"""
    _save_step_quantities!(integ, i)

Save the per-step quantities of the step that just finished: its diffusion estimate and,
if they are consumed later, the backward transition kernel and the smoother state.

All three are one-per-step and hence indexed by the *transition* `sol.t[i-1] -> sol.t[i]`,
i.e. `length(...) == length(sol.t) - 1`; `i` is the index of the save point that the step
ended in. Called from `savevalues!` for every saved step, and from
`pn_solution_endpoint_match_cur_integrator!` for a final step that only the endpoint match
saved.
"""
function _save_step_quantities!(integ, i)
    save_diffusion!(integ.sol, i, integ.cache.local_diffusion)
    # Only stored when consumed; see `_needs_backward_kernels`.
    if _needs_backward_kernels(integ.alg)
        copyat_or_push!(integ.sol.backward_kernels, i, integ.cache.backward_kernel)
    end
    if integ.alg.smooth && integ.alg.smoother == :mbf
        _save_smoother_state!(integ.sol.smoother_states, integ.cache)
    end
    return nothing
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
    # `_savevalues!` returns `(saved, savedexactly)`; `saved` is `true` exactly when it
    # appended a new entry to `sol.t`. Without a save there is no solution index that the
    # current state belongs to, so the custom saves below have to be skipped: otherwise
    # they overwrite the last saved entry with the current (unsaved) state. This happens
    # e.g. with `save_end=false`, where the final step is taken but never saved.
    saved, _ = out

    # Save our custom stuff that we need for the posterior
    if saved && integ.opts.save_everystep
        i = integ.saveiter
        copyat_or_push!(integ.sol.x_filt, i, integ.cache.x)
        _gaussian_mul!(integ.cache.pu_tmp, integ.cache.SolProj, integ.cache.x)
        copyat_or_push!(integ.sol.pu, i, integ.cache.pu_tmp)

        # The per-step quantities (diffusion, backward kernel, smoother state) are indexed
        # by the transition sol.t[j] -> sol.t[j+1], so the step just taken (ending at the
        # freshly saved point i) is appended at i - 1.
        _save_step_quantities!(integ, i)
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
