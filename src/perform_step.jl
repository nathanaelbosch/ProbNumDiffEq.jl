# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ::OrdinaryDiffEq.ODEIntegrator, cache::EKCache)
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
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    initial_pu = _gaussian_mul!(cache.pu_tmp, cache.SolProj, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, initial_pu)

    return nothing
end

function make_new_transitions(integ, cache, repeat_step)::Bool
    # Similar to OrdinaryDiffEq.do_newJ
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

As in OrdinaryDiffEq.jl, this step is not necessarily successful!
For that functionality, use `OrdinaryDiffEq.step!(integ)`.
"""
function OrdinaryDiffEq.perform_step!(integ, cache::EKCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, SolProj = integ.cache
    @unpack xprev, x_pred, u_pred, x_filt, err_tmp = integ.cache
    @unpack A, Q, Ah, Qh, P, PI = integ.cache

    tnew = t + dt

    if make_new_transitions(integ, cache, repeat_step)
        # Rosenbrock-style update of the IOUP rate parameter
        if cache.prior isa IOUP && cache.prior.update_rate_parameter
            OrdinaryDiffEq.calc_J!(cache.prior.rate_parameter, integ, cache, false)
        end

        make_transition_matrices!(cache, cache.prior, dt)
    end

    # Predict the mean
    predict_mean!(x_pred.μ, xprev.μ, Ah)
    write_into_solution!(
        integ.u, x_pred.μ; cache, is_secondorder_ode=integ.f isa DynamicalODEFunction)

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
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))
    # integ.sol.log_likelihood += integ.cache.log_likelihood

    # Update state and save the ODE solution value
    x_filt = update!(cache, x_pred)
    write_into_solution!(
        integ.u, x_filt.μ; cache, is_secondorder_ode=integ.f isa DynamicalODEFunction)

    # Update the global diffusion MLE (if applicable)
    if !isdynamic(cache.diffusionmodel)
        cache.global_diffusion = estimate_global_diffusion(cache.diffusionmodel, integ)
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
as in OrdinaryDiffEq.jl.
"""
function evaluate_ode!(integ, x_pred, t)
    @unpack f, p, dt = integ
    @unpack du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)

    z = integ.cache.measurement
    z_tmp = integ.cache.m_tmp

    integ.cache.measurement_model(z.μ, x_pred.μ, p, t)
    integ.stats.nf += 1

    calc_H!(H, integ, integ.cache)

    return nothing
end

compute_measurement_covariance!(cache) =
    fast_X_A_Xt!(cache.measurement.Σ, cache.x_pred.Σ, cache.H)

function update!(cache, prediction)
    @unpack measurement, H, x_filt, K1, m_tmp, C_DxD = cache
    @unpack C_dxd, C_Dxd = cache
    K2 = C_Dxd
    update!(x_filt, prediction, measurement, H, K1, K2, C_DxD, C_dxd)
    return x_filt
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
            integ.u[1, :],
            integ.uprev[1, :],
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
    @unpack local_diffusion, Qh, H, d = cache

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    R = cache.measurement.Σ.R

    if local_diffusion isa Diagonal
        _QR = cache.C_DxD .= Qh.R .* sqrt.(local_diffusion.diag)'
        _matmul!(R, _QR, H')
        error_estimate = view(cache.tmp, 1:d)
        sum!(abs2, error_estimate', view(R, :, 1:d))
        error_estimate .= sqrt.(error_estimate)
        return error_estimate
    elseif local_diffusion isa Number
        _matmul!(R, Qh.R, H')

        # error_estimate = diag(PSDMatrix(R))
        # error_estimate .*= local_diffusion
        # error_estimate .= sqrt.(error_estimate)
        # error_estimate = view(error_estimate, 1:d)

        # faster:
        error_estimate = view(cache.tmp, 1:d)
        if R isa Kronecker.AbstractKroneckerProduct
            error_estimate .= sum(abs2, R.B)
        else
            sum!(abs2, error_estimate', view(R, :, 1:d))
        end
        error_estimate .*= local_diffusion
        error_estimate .= sqrt.(error_estimate)

        return error_estimate
    end
end
