# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
end

"""Perform a step

Not necessarily successful! For that, see `step!(integ)`.

Basically consists of the following steps
- Coordinate change / Predonditioning
- Prediction step
- Measurement: Evaluate f and Jf; Build z, S, H
- Calibration; Adjust prediction / measurement covs if the diffusion model "dynamic"
- Update step
- Error estimation
- Undo the coordinate change / Predonditioning
"""
function OrdinaryDiffEq.perform_step!(integ, cache::GaussianODEFilterCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache

    tnew = t + dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = inv(P)
    x = P * x

    if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

        # Predict
        predict_mean!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict_cov!(x_pred, x, A, apply_diffusion(Q, integ.cache.diffusion))
        copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)
        measure!(integ, x_pred, tnew)
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)

    end

    # Likelihood
    cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)
    mul!(u_filt, SolProj, PI*x_filt.μ)
    integ.u .= u_filt

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u, u_filt,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar

    end
    # stuff that would normally be in apply_step!
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end

function measure!(integ, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
    @assert iszero(R)

    PI = inv(Precond(dt))
    E0, E1 = Proj(0), Proj(1)

    z, S = measurement.μ, measurement.Σ

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1
    z .= E1*PI*x_pred.μ .- du

    # Cov
    if alg isa EK1 || alg isa IEKS
        linearize_at = (alg isa IEKS && !isnothing(alg.linearize_at)) ?
            alg.linearize_at(t).μ : u_pred
        _eval_f_jac!(ddu, linearize_at, p, t, f)
        integ.destats.njacs += 1
        mul!(H, (E1 .- ddu * E0), PI)
    else
        mul!(H, E1, PI)
    end
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return measurement
end

# The following functions are just there to handle both IIP and OOP easily
_eval_f!(du, u, p, t, f::AbstractODEFunction{true}) = f(du, u, p, t)
_eval_f!(du, u, p, t, f::AbstractODEFunction{false}) = (du .= f(u, p, t))
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{true}) = f.jac(ddu, u, p, t)
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{false}) = (ddu .= f.jac(u, p, t))

function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    update!(x_filt, prediction, measurement, H, R)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack diffusion, Q, H = integ.cache

    if diffusion isa Real && isinf(diffusion)
        return Inf
    end

    error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), H))))






    # Try out some wild things on embedded error estimation
    @unpack dt = integ
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, x_filt, u_filt = integ.cache
    @unpack A, Q = integ.cache
    q = integ.alg.order

    P = Precond(dt)
    PI = inv(P)
    # x = P*x
    E0 = Proj(0)
    Ah = PI * A * P
    Qh = X_A_Xt(Q, PI)

    # Just to test: re-create the prediction step in here
    x_tmp = copy(x_pred)
    predict!(x_tmp, x, Ah, apply_diffusion(Qh, integ.cache.diffusion))
    @assert x_tmp ≈ x_pred

    # Now do the thing with a lower order
    m_lower = x.μ[1:d*q]
    P_lower_L = collect(qr(x.Σ.squareroot[1:d*q, :]').R')
    P_lower = SquarerootMatrix(P_lower_L)
    x_lower = Gaussian(m_lower, P_lower)

    A_lower, Q_lower = ibm(d, q-1)
    Precond_lower = preconditioner(d, q-1)
    P_l = Precond_lower(dt)
    PI_l = inv(P_l)
    Ah_lower = PI_l * A_lower * P_l
    Qh_lower = X_A_Xt(Q_lower, PI_l)

    x_pred_lower = copy(x_lower)
    predict!(x_pred_lower, x_lower, Ah_lower, apply_diffusion(Qh_lower, integ.cache.diffusion))
    # @info "predict with lower order" x_pred_lower.μ x_pred.μ



    # measure
    @unpack f, p, dt, alg, t = integ
    @unpack u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
    E0 = Proj(0)[:, 1:d*q]
    E1 = Proj(1)[:, 1:d*q]
    _m = copy(measurement)
    z, S = _m.μ, _m.Σ
    _eval_f!(du, E0 * x_pred_lower.μ, p, t+dt, f)
    integ.destats.nf += 1
    z .= E1*x_pred_lower.μ .- du

    @assert !(alg isa IEKS)
    H = copy(H)[:, 1:d*q]
    if alg isa EK1
        _eval_f_jac!(ddu, E0*x_pred_lower.μ, p, t+dt, f)
        integ.destats.njacs += 1
        H .= E1 .- ddu * E0
    else
        H .= E1
    end
    copy!(S, Matrix(X_A_Xt(x_pred_lower.Σ, H)))

    # update
    x_filt_lower = copy(x_pred_lower)
    update!(x_filt_lower, x_pred_lower, _m, H, R)



    # Finally: Compare the orders!
    # @info "Estimate_errors" E0*x_filt_lower.μ - integ.cache.u_filt
    error_estimate = E0*x_filt_lower.μ - integ.cache.u_filt


    # @info "estimate_errors" x.μ
    return error_estimate
end
