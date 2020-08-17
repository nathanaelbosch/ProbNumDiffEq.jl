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
    @unpack state_estimates, times = integ
    proposals = integ.proposals
    accepted_proposals = [p for p in proposals if p.accept]
    for i in length(state_estimates)-1:-1:2
        h = accepted_proposals[i].dt  # step t -> t+1
        h2 = times[i+1] - times[i]
        @assert h ≈ h2

        @assert accepted_proposals[i].t == times[i+1]

        prediction = accepted_proposals[i].prediction  # t+1
        filter_estimate = state_estimates[i]  # t
        smoothed_estimate = state_estimates[i+1] # t+1

        A = integ.dm.A(h)
        Q = integ.dm.Q(h)
        state_estimates[i] = smooth(filter_estimate, prediction, smoothed_estimate, (A=A, Q=Q))
    end
end

function calibrate!(integ::ODEFilterIntegrator)
    @unpack state_estimates = integ
    proposals = integ.proposals
    σ² = static_sigma_estimation(integ.sigma_estimator, integ, proposals)
    if σ² != 1
        for s in state_estimates
            s.Σ .*= σ²
        end
        for p in proposals
            p.measurement.Σ .*= σ²
            p.prediction.Σ .*= σ²
            p.filter_estimate.Σ .*= σ²
        end
    end
end
