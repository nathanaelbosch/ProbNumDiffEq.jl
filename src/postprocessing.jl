########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################

function smooth(filter_estimate::Gaussian,
                prediction::Gaussian,
                smoothed_estimate::Gaussian,
                dynamics_model)

    m, P = kf_smooth(
        filter_estimate.μ, filter_estimate.Σ,
        prediction.μ, prediction.Σ,
        smoothed_estimate.μ, smoothed_estimate.Σ,
        dynamics_model.A, dynamics_model.Q
    )

    return Gaussian(m, P)
end

function smooth!(integ::ODEFilterIntegrator)

    @unpack state_estimates, times, sigmas = integ
    @unpack A!, Q! = integ.constants
    @unpack Ah, Qh, x_pred = integ.cache

    for i in length(state_estimates)-1:-1:2
        h = times[i+1] - times[i]

        filter_estimate = state_estimates[i]  # t

        A!(Ah, h)
        Q!(Qh, h, sigmas[i])
        mul!(x_pred.μ, Ah, filter_estimate.μ)
        x_pred.Σ .= Ah * filter_estimate.Σ * Ah' .+ Qh

        smoothed_estimate = state_estimates[i+1] # t+1

        state_estimates[i] = smooth(filter_estimate, x_pred, smoothed_estimate, (A=Ah, Q=Qh))
    end
end

function calibrate!(integ::ODEFilterIntegrator)

    @unpack state_estimates = integ
    proposals = integ.proposals
    σ² = static_sigma_estimation(integ.sigma_estimator, integ, proposals)
    if σ² != 1
        @assert all(integ.sigmas .== 1) "Currently, sigma has to bei EITHER dynamics OR fixed, but this seems like a mix of both!"
        for s in state_estimates
            s.Σ .*= σ²
        end
        integ.sigmas .= σ²
        # for p in proposals
        #     p.measurement.Σ .*= σ²
        #     p.prediction.Σ .*= σ²
        #     p.filter_estimate.Σ .*= σ²
        # end
    end
end
