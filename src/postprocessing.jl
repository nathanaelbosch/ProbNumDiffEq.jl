########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################

function smooth(filter_estimate::Gaussian,
                prediction::Gaussian,
                smoothed_estimate::Gaussian,
                dynamics_model,
                precond)

    m, P = kf_smooth(
        filter_estimate.μ, filter_estimate.Σ,
        prediction.μ, prediction.Σ,
        smoothed_estimate.μ, smoothed_estimate.Σ,
        dynamics_model.A, dynamics_model.Q,
        precond,
    )

    return Gaussian(m, P)
end

function smooth!(integ::ODEFilterIntegrator)

    @unpack state_estimates, times, sigmas = integ
    @unpack A!, Q! = integ.constants
    @unpack Ah, Qh, x_pred = integ.cache
    # x_pred is just used as a cache here

    for i in length(state_estimates)-1:-1:2
        h = times[i+1] - times[i]

        filter_estimate = state_estimates[i]  # t

        A!(Ah, h)
        Q!(Qh, h, sigmas[i])
        # Prediction: t -> t+1
        mul!(x_pred.μ, Ah, filter_estimate.μ)
        x_pred.Σ .= Ah * filter_estimate.Σ * Ah' .+ Qh

        smoothed_estimate = state_estimates[i+1] # t+1
        state_estimates[i] = smooth(filter_estimate, x_pred, smoothed_estimate,
                                    (A=Ah, Q=Qh),
                                    integ.constants.Precond)
    end
end

function calibrate!(integ::ODEFilterIntegrator)

    @unpack state_estimates, sigmas = integ
    for s in state_estimates
        s.Σ .*= sigmas[end]
    end
end
