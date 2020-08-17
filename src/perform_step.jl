"""Perform a step, but not necessarily successful!

This is the actual interestin part of the algorithm
"""
function perform_step!(integ::ODEFilterIntegrator)

    integ.iter += 1
    t = integ.t + integ.dt
    integ.t_new = t

    x_pred = predict!(integ)
    integ.cache.u_pred .= integ.constants.E0 * x_pred.μ

    h = measure_h!(integ, x_pred, t)
    H = measure_H!(integ, x_pred, t)

    σ_sq = dynamic_sigma_estimation(integ.sigma_estimator, integ)
    x_pred.Σ .+= (σ_sq - 1) .* integ.cache.Qh
    integ.cache.σ_sq = σ_sq

    x_filt = update!(integ, x_pred)

    # integ.EEst = 0
end


function predict!(integ::ODEFilterIntegrator)

    @unpack dt = integ
    @unpack A!, Q! = integ.constants
    @unpack x, Ah, Qh, x_pred = integ.cache

    A!(Ah, dt)
    Q!(Qh, dt)

    # x_pred.μ .= Ah * x.μ
    mul!(x_pred.μ, Ah, x.μ)
    x_pred.Σ .= Ah * x.Σ * Ah' .+ Qh
    return x_pred
end


function measure_h!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack t_new, p, f = integ
    @unpack E0, h! = integ.constants
    @unpack du, h, u_pred = integ.cache

    IIP = isinplace(integ)
    if IIP
        f(du, u_pred, p, t_new)
    else
        du .= f(u_pred, p, t_new)
    end
    integ.destats.nf += 1

    h!(h, du, x_pred.μ)

    return h
end

function measure_H!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, t_new, f = integ
    @unpack jac, H! = integ.constants
    @unpack u_pred, ddu, H = integ.cache

    if !isnothing(jac)
        if isinplace(integ)
            jac(ddu, u_pred, p, t_new)
        else
            ddu .= jac(u_pred, p, t_new)
        end
        H!(H, ddu)
        integ.destats.njacs += 1
    else
        H!(H, ddu)
    end


    return H
end

function update!(integ::ODEFilterIntegrator, prediction)

    @unpack R = integ.constants
    @unpack measurement, h, H, K, x_filt = integ.cache
    v, S = measurement.μ, measurement.Σ
    v .= 0 .- h

    m_p, P_p = prediction.μ, prediction.Σ
    S .= H * P_p * H' .+ R
    S_inv = inv(S)
    K .= P_p * H' * S_inv

    x_filt.μ .= m_p .+ K*v
    x_filt.Σ .= P_p .- K*S*K'

    return x_filt, measurement
end
