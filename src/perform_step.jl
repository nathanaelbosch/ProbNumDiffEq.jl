# Called in the OrdinaryDiffEqCore.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEqCore.initialize!(
    integ::OrdinaryDiffEqCore.ODEIntegrator,
    cache::EKCache,
)
    check_secondorderode(integ)
    check_densesmooth(integ)
    check_saveiter(integ)

    integ.kshortsize = 1
    resize!(integ.k, integ.kshortsize)
    integ.k[1] = integ.u

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ, cache)
    copy!(integ.cache.xprev, integ.cache.x)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEqCore.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    initial_pu = _gaussian_mul!(cache.pu_tmp, cache.SolProj, cache.x)
    OrdinaryDiffEqCore.copyat_or_push!(integ.sol.pu, integ.saveiter, initial_pu)

    return nothing
end

function make_new_transitions(integ, cache, repeat_step)::Bool
    # Similar to OrdinaryDiffEqCore.do_newJ
    if integ.iter <= 1
        return true
    elseif cache.prior isa IOUP && cache.prior.update_rate_parameter
        return true
    elseif repeat_step
        return false
    elseif integ.dt == cache.dt_last
        return false
    else
        return true
    end
end

function write_into_solution!(u, μ; cache::EKCache, is_secondorder_ode=false)
    if is_secondorder_ode
        _matmul!(view(u.x[2], :), cache.E0, μ)
        _matmul!(view(u.x[1], :), cache.E1, μ)
    else
        _matmul!(view(u, :), cache.E0, μ)
    end
end

"""
    perform_step!(integ, cache::EKCache[, repeat_step=false])

Perform the ODE filter step.

Basically consists of the following steps
- Compute the current transition and diffusion matrices
- Predict mean
- Evaluate the ODE (and Jacobian) at the predicted mean; Build measurement mean `z`
- Compute local diffusion and local error estimate
- If the step is rejected, terminate here; Else continue
- Predict the covariance and build the measurement covariance `S`
- Kalman update step
- (optional) Update the global diffusion MLE

As in OrdinaryDiffEqCore.jl, this step is not necessarily successful!
For that functionality, use `OrdinaryDiffEqCore.step!(integ)`.
"""
function OrdinaryDiffEqCore.perform_step!(integ, cache::EKCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d = integ.cache
    @unpack xprev, x_pred, u_pred, x_filt, err_tmp = integ.cache
    @unpack A, Q, Ah, Qh, P, PI = integ.cache

    tnew = t + dt

    if make_new_transitions(integ, cache, repeat_step)
        # Rosenbrock-style update of the IOUP rate parameter
        if cache.prior isa IOUP && cache.prior.update_rate_parameter
            OrdinaryDiffEqDifferentiation.calc_J!(
                cache.prior.rate_parameter,
                integ,
                cache,
                false,
            )
        end

        make_transition_matrices!(cache, cache.prior, dt)
    end

    # Predict the mean
    predict_mean!(x_pred.μ, xprev.μ, Ah)
    write_into_solution!(
        integ.u, x_pred.μ; cache, is_secondorder_ode=(integ.f isa DynamicalODEFunction))

    # Measure
    evaluate_ode!(integ, x_pred, tnew)

    # Estimate diffusion, and (if adaptive) the local error estimate; Stop here if rejected
    if integ.opts.adaptive || isdynamic(cache.diffusionmodel)
        cache.local_diffusion = estimate_local_diffusion(cache.diffusionmodel, integ)
    end
    if integ.opts.adaptive
        integ.EEst = compute_scaled_error_estimate!(integ, cache)
        if integ.EEst >= one(integ.EEst)
            return
        end
    end

    # Predict the covariance, using either the local or global diffusion
    extrapolation_diff =
        isdynamic(cache.diffusionmodel) ? cache.local_diffusion : cache.default_diffusion
    predict_cov!(x_pred.Σ, xprev.Σ, Ah, Qh, cache.C_DxD, cache.C_2DxD, extrapolation_diff)

    if integ.alg.smooth
        @unpack C_DxD, backward_kernel = cache
        K = AffineNormalKernel(Ah, Qh)
        compute_backward_kernel!(
            backward_kernel, x_pred, xprev, K; C_DxD, diffusion=extrapolation_diff)
    end

    # Compute measurement covariance only now; likelihood computation is currently broken
    compute_measurement_covariance!(cache)

    # Update state and save the ODE solution value
    x_filt, loglikelihood = update!(
        x_filt, x_pred, cache.measurement, cache.H; cache, R=cache.R)
    write_into_solution!(
        integ.u, x_filt.μ; cache, is_secondorder_ode=(integ.f isa DynamicalODEFunction))

    cache.log_likelihood = loglikelihood
    integ.sol.pnstats.log_likelihood += cache.log_likelihood

    # Update the global diffusion MLE (if applicable)
    if !isdynamic(cache.diffusionmodel)
        estimate_global_diffusion(cache.diffusionmodel, integ)
    end

    # Advance the state
    copy!(integ.cache.x, integ.cache.x_filt)

    return nothing
end

"""
    evaluate_ode!(integ, x_pred, t)

Evaluate the ODE vector field and, if using the [`EK1`](@ref), its Jacobian.

In addition, compute the measurement mean (`z`) and the measurement function Jacobian (`H`).
Results are saved into `integ.cache.du`, `integ.cache.ddu`, `integ.cache.measurement.μ`
and `integ.cache.H`.
Jacobians are computed either with the supplied `f.jac`, or via automatic differentiation,
as in OrdinaryDiffEqCore.jl.
"""
function evaluate_ode!(integ, x_pred, t)
    @unpack f, p, dt = integ
    @unpack du, ddu, measurement, R, H = integ.cache

    z = integ.cache.measurement
    z_tmp = integ.cache.m_tmp

    integ.cache.measurement_model(z.μ, x_pred.μ, p, t)
    integ.stats.nf += 1

    calc_H!(H, integ, integ.cache)

    return nothing
end

compute_measurement_covariance!(cache) = begin
    _matmul!(cache.C_Dxd, cache.x_pred.Σ.R, cache.H')
    _matmul!(cache.measurement.Σ, cache.C_Dxd', cache.C_Dxd)
    if !isnothing(cache.R)
        add!(cache.measurement.Σ, _matmul!(cache.C_dxd, cache.R.R', cache.R.R))
    end
end

"""
    compute_scaled_error_estimate!(integ, cache)

Compute the scaled, local error estimate `Eest`, that should satisfy `Eest < 1`.
The actual local error is computed with [`estimate_errors!`](@ref).
Then, `DiffEqBase.calculate_residuals!` handles the scaling with adaptive and relative
tolerances, and `integ.opts.internalnorm` provides the norm that should be used to return
only a scalar.
"""
function compute_scaled_error_estimate!(integ, cache)
    @unpack err_tmp = cache
    t = integ.t + integ.dt
    err_est_unscaled = estimate_errors!(cache)
    err_est_unscaled .*= integ.dt
    if integ.f isa DynamicalODEFunction # second-order ODE
        DiffEqBase.calculate_residuals!(
            err_tmp,
            err_est_unscaled,
            integ.u.x[1],
            integ.uprev.x[1],
            integ.opts.abstol,
            integ.opts.reltol,
            integ.opts.internalnorm,
            t,
        )
    else # regular first-order ODE
        DiffEqBase.calculate_residuals!(
            err_tmp,
            err_est_unscaled,
            integ.u,
            integ.uprev,
            integ.opts.abstol,
            integ.opts.reltol,
            integ.opts.internalnorm,
            t,
        )
    end
    return integ.opts.internalnorm(err_tmp, t) # scalar
end

"""
    estimate_errors!(cache)

Computes a local error estimate, as
```math
E_i = ( σ_{loc}^2 ⋅ (H Q(h) H^T)_{ii} )^(1/2)
```
To save allocations, the function modifies the given `cache` and writes into
`cache.C_Dxd` during some computations.
"""
function estimate_errors!(cache::AbstractODEFilterCache)
    @unpack local_diffusion, Qh, H, C_d, C_Dxd, C_DxD = cache
    _Q = apply_diffusion!(PSDMatrix(C_DxD), Qh, local_diffusion)
    _HQH = PSDMatrix(_matmul!(C_Dxd, _Q.R, H'))
    error_estimate = diag!(C_d, _HQH)
    @.. error_estimate = sqrt(error_estimate)
    return error_estimate
end

diag!(v::AbstractVector, M::PSDMatrix) = (sum!(abs2, v', M.R); v)
diag!(v::AbstractVector, M::PSDMatrix{<:Number,<:IsometricKroneckerProduct}) =
    v .= sum(abs2, M.R.B)
diag!(v::AbstractVector, M::PSDMatrix{<:Number,<:BlocksOfDiagonals}) = begin
    @assert length(v) == nblocks(M.R)
    @assert size(blocks(M.R)[1], 2) == 1 # assumes all of them have the same shape
    @simd ivdep for i in eachindex(blocks(M.R))
        v[i] = sum(abs2, blocks(M.R)[i])
    end
    return v
end
