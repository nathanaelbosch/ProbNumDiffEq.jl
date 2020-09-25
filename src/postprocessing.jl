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
        Q!(Qh, dt)
        Qh .*= sigmas[i]

        # @info "Smoothing" i dt sigmas[i] Qh

        # @info "smooth_all!" state_estimates[i].Σ state_estimates[i+1].Σ
        # @info "smooth_all!" P*state_estimates[i].Σ P*state_estimates[i+1].Σ
        state_estimates[i] = P * state_estimates[i]
        smooth!(state_estimates[i], P*state_estimates[i+1], Ah, Qh, integ, PI)
        any(isnan.(state_estimates[i].μ)) && error("NaN mean after smoothing")
        any(isnan.(state_estimates[i].Σ)) && error("NaN cov after smoothing")
        state_estimates[i] = PI * state_estimates[i]
    end
end


function smooth!(x_curr, x_next, Ah, Qh, integ, PI=I)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing

    @unpack d, q = integ.constants
    @unpack x_tmp = integ.cache

    # @info "smooth!" x_curr.Σ x_next.Σ Ah Qh PI
    if all((Qh) .< eps(eltype(Qh)))
        @warn "smooth: Qh is really small! The system is basically deterministic, so we just \"predict backwards\"."
        return inv(Ah) * x_next
    end


    # Prediction: t -> t+1
    mul!(x_tmp.μ, Ah, x_curr.μ)
    x_tmp.Σ .= Ah * x_curr.Σ * Ah' .+ Qh


    # Smoothing
    cov_before = copy(x_curr.Σ)
    cov_pred = copy(x_tmp.Σ)
    P_p = Symmetric(cov_pred)
    P_p_inv = inv(P_p)
    G = x_curr.Σ * Ah' * P_p_inv
    x_curr.μ .+= G * (x_next.μ .- x_tmp.μ)

    # Vanilla:
    cov_diff = x_next.Σ .- x_tmp.Σ
    zero_if_approx_similar!(cov_diff, x_next.Σ, x_tmp.Σ)
    GDG = G * cov_diff * G'
    # GDG = x_curr.Σ * Ah' * (P_p \ (
    #     x_curr.Σ * Ah' * (P_p \ cov_diff')
    # )')
    x_tmp.Σ .= x_curr.Σ .+ GDG
    zero_if_approx_similar!(x_tmp.Σ, x_curr.Σ, -GDG)
    zero_if_approx_similar!(x_tmp.Σ, PI*x_curr.Σ*PI', -PI*GDG*PI')
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
        @info "Error while smoothing: negative variances! (in x_curr)" cov_before cov_pred x_next.Σ x_curr.Σ Qh GDG
        throw(e)
    end


    return nothing
end
