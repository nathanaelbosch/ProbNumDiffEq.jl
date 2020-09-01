"""Perform a step, but not necessarily successful!

This is the actual interestin part of the algorithm
"""
function perform_step!(integ::ODEFilterIntegrator)
    @unpack t, dt = integ
    @unpack E0 = integ.constants
    @unpack x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache

    t = t + dt
    integ.t_new = t

    x_pred = predict!(integ)
    mul!(u_pred, E0, x_pred.μ)

    measure_h!(integ, x_pred, t)
    measure_H!(integ, x_pred, t)

    if isdynamic(integ.sigma_estimator)
        σ_sq = dynamic_sigma_estimation(integ.sigma_estimator, integ)
        x_pred.Σ .+= (σ_sq - 1) .* integ.cache.Qh
        integ.cache.σ_sq = σ_sq
    end

    x_filt = update!(integ, x_pred)
    mul!(u_filt, E0, x_filt.μ)

    if isstatic(integ.sigma_estimator)
        # E.g. estimate the /current/ MLE sigma
        σ_sq = static_sigma_estimation(integ.sigma_estimator, integ)
        integ.cache.σ_sq = σ_sq
    end

    err_est_unscaled = estimate_errors(integ.error_estimator, integ)
    # Scale the error with old u-values and tolerances
    DiffEqBase.calculate_residuals!(
        err_tmp,
        dt * err_est_unscaled, integ.u, u_filt, integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
    err_est_combined = integ.opts.internalnorm(err_tmp, t)  # Norm over the dimensions
    integ.EEst = err_est_combined

end


function predict!(integ::ODEFilterIntegrator)

    @unpack dt = integ
    @unpack A!, Q! = integ.constants
    @unpack x, Ah, Qh, x_pred = integ.cache

    A!(Ah, dt)
    Q!(Qh, dt)

    # x_pred.μ .= Ah * x.μ
    mul!(x_pred.μ, Ah, x.μ)
    x_pred.Σ .= Symmetric(Ah * x.Σ * Ah' .+ Qh)
    return x_pred
end


function measure_h!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, f = integ
    @unpack E0, h! = integ.constants
    @unpack du, h, u_pred = integ.cache

    IIP = isinplace(integ)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    h!(h, du, x_pred.μ)
end

function measure_H!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, f = integ
    @unpack jac, H! = integ.constants
    @unpack u_pred, ddu, H = integ.cache

    if !isnothing(jac)
        if isinplace(integ)
            jac(ddu, u_pred, p, t)
        else
            ddu .= jac(u_pred, p, t)
        end
        integ.destats.njacs += 1
    end
    H!(H, ddu)
end

function update!(integ::ODEFilterIntegrator, prediction)

    @unpack R = integ.constants
    @unpack measurement, h, H, K, x_filt = integ.cache
    v, S = measurement.μ, measurement.Σ
    v .= 0 .- h

    m_p, P_p = prediction.μ, prediction.Σ
    S .= Symmetric(H * P_p * H' .+ R)
    S_inv = inv(S)
    K .= P_p * H' * S_inv

    x_filt.μ .= m_p .+ K*v
    x_filt.Σ .= P_p .- Symmetric(K*S*K')

    return x_filt
end
