"""Perform a step, but not necessarily successful!

This is the actual interestin part of the algorithm
"""
function perform_step!(integ::ODEFilterIntegrator)
    integ.iter += 1
    t = integ.t + integ.dt
    prediction, A, Q = predict(integ)
    h, H = measure(integ, prediction, t)

    σ_sq = dynamic_sigma_estimation(
        integ.sigma_estimator; prediction=prediction, v=h, H=H, Q=Q)
    prediction = Gaussian(prediction.μ, prediction.Σ + (σ_sq - 1) * Q)

    filter_estimate, measurement = update(integ, prediction, h, H)

    proposal = (t=t,
            prediction=prediction,
            filter_estimate=filter_estimate,
            measurement=measurement,
            H=H, Q=Q, v=h,
            σ²=σ_sq)

    integ.proposal = proposal

    # integ.EEst = 0
end


function predict(integ)
    @unpack dm, mm, x, t, dt = integ

    m, P = x.μ, x.Σ
    A, Q = dm.A(dt), dm.Q(dt)
    m_p = A * m
    P_p = Symmetric(A*P*A') + Q
    prediction=Gaussian(m_p, P_p)
    return prediction, A, Q
end


function measure(integ, prediction, t)
    @unpack mm = integ
    m_p = prediction.μ
    h = mm.h(m_p, t)
    H = mm.H(m_p, t)
    return h, H
end


function update(integ, prediction, h, H)
    R = integ.mm.R
    v = 0 .- h

    m_p, P_p = prediction.μ, prediction.Σ
    S = Symmetric(H * P_p * H' + R)
    K = P_p * H' * inv(S)
    m = m_p + K*v
    P = P_p - Symmetric(K*S*K')
    return Gaussian(m, P), Gaussian(v, S)
end
