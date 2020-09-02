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
        smooth!(filter_estimate, x_pred, smoothed_estimate, integ)
    end
end

function smooth!(x_filt, x_pred, x_smooth, integ)
    @unpack Ah = integ.cache
    @unpack d, q, Precond = integ.constants

    P = copy(x_filt.Σ)
    G = x_filt.Σ * Ah' * inv(x_pred.Σ)
    x_filt.μ .+= G * (x_smooth.μ .- x_pred.μ)
    x_filt.Σ .+= G * (x_smooth.Σ .- x_pred.Σ) * G'

    # Sanity: Make sure that the diagonal of P is non-negative
    minval = minimum(diag(x_filt.Σ))
    if abs(minval) < eps(eltype(x_filt.Σ))
    # if abs(minval) < 1e-12
        for i in 1:(d*(q+1))
            x_filt.Σ[i,i] -= minval
        end
    end
    minval = minimum(diag(x_filt.Σ))
    if minval < 0
        @info "Error while smoothing: negative variances!" x_filt x_pred x_smooth minval eps(eltype(x_filt.Σ))
        display(P)
        display(x_pred.Σ)
        display(x_smooth.Σ)
        display(x_filt.Σ)
        display(G)
        @info "Solver used" integ.constants.q integ.sigma_estimator integ.steprule integ.smooth
        error("Encountered negative variances during smoothing")
    end
    @assert all(diag(x_filt.Σ) .>= 0) "The covariance `P` might be NaN! Make sure that the covariances during the solve make sense."

    return nothing
end



function calibrate!(integ::ODEFilterIntegrator)

    @unpack state_estimates, sigmas = integ
    for s in state_estimates
        s.Σ .*= sigmas[end]
    end
end
