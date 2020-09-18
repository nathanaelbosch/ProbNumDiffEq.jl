########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################
function calibrate!(integ::ODEFilterIntegrator)

    @unpack state_estimates, sigmas = integ
    for s in state_estimates
        s.Σ .*= sigmas[end]
    end
end




function smooth_all!(integ::ODEFilterIntegrator)

    @unpack state_estimates, times, sigmas = integ
    @unpack A!, Q!, Precond, InvPrecond = integ.constants
    @unpack Ah, Qh = integ.cache
    # x_pred is just used as a cache here

    for i in length(state_estimates)-1:-1:2
        dt = times[i+1] - times[i]

        P = Precond(dt)
        PI = InvPrecond(dt)

        A!(Ah, dt)
        Q!(Qh, dt, sigmas[i])

        # @info "Smoothing" i dt sigmas[i] sigmas[i]==0

        state_estimates[i] = P * state_estimates[i]
        smooth!(state_estimates[i], P*state_estimates[i+1], Ah, Qh, integ)
        state_estimates[i] = PI * state_estimates[i]
    end
end


function smooth!(x_curr, x_next, Ah, Qh, integ)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing

    @unpack d, q = integ.constants
    @unpack x_tmp = integ.cache
    x_pred = x_tmp

    if all((Qh) .< eps(eltype(Qh)))
        @warn "smooth: Qh is really small! The system is basically deterministic, so we just \"predict backwards\"."
        return inv(Ah) * x_next
    end


    # Prediction: t -> t+1
    mul!(x_pred.μ, Ah, x_curr.μ)
    x_pred.Σ .= Ah * x_curr.Σ * Ah' .+ Qh


    # Smoothing
    P = copy(x_curr.Σ)
    try
        inv(Symmetric(x_pred.Σ))
    catch
        @warn "Inverse not working" x_pred.Σ x_curr.Σ x_next.Σ
        IP = integ.constants.InvPrecond
        @info "Without the preconditioning:" IP*x_pred.Σ*IP' IP*x_curr.Σ*IP' IP*x_next.Σ*IP'
    end
    P_p_inv = inv(Symmetric(x_pred.Σ))
    G = x_curr.Σ * Ah' * P_p_inv
    x_curr.μ .+= G * (x_next.μ .- x_pred.μ)

    # Vanilla:
    GDG = G * (x_next.Σ .- x_pred.Σ) * G'
    x_tmp.Σ .= x_curr.Σ .+ GDG
    zero_if_approx_similar!(x_tmp.Σ, x_curr.Σ, -GDG)
    copy!(x_curr.Σ, x_tmp.Σ)
    # Joseph-Form:
    # P = copy(x_curr.Σ)
    # C_tilde = Ah
    # K_tilde = P * Ah' * P_p_inv
    # P_s = ((I - K_tilde*C_tilde) * P * (I - K_tilde*C_tilde)'
    #        + K_tilde * Qh * K_tilde' + G * x_next.Σ * G')
    # x_curr.Σ .= P_s


    try
        assert_good_covariance(x_curr.Σ)
    catch e
        @info "Error while smoothing: negative variances! (in x_curr)" P x_pred.Σ x_next.Σ x_curr.Σ Ah Qh G GDG
        @info "Solver used" integ.constants.q integ.sigma_estimator integ.steprule integ.smooth
        throw(e)
    end


    return nothing
end
