########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################
function smooth_all!(integ)

    @unpack x, t, diffusions = integ.sol
    @unpack A!, Q!, Precond, InvPrecond = integ.cache
    @unpack Ah, Qh = integ.cache
    # x_pred is just used as a cache here

    for i in length(x)-1:-1:2
        dt = t[i+1] - t[i]

        P = Precond(dt)
        PI = InvPrecond(dt)

        A!(Ah, dt)
        Q!(Qh, dt)
        Qh .*= diffusions[i]

        # @info "Smoothing" i dt diffusions[i] Qh

        # @info "smooth_all!" state_estimates[i].Σ state_estimates[i+1].Σ
        # @info "smooth_all!" P*state_estimates[i].Σ P*state_estimates[i+1].Σ
        x[i] = P * x[i]
        smooth!(x[i], P*x[i+1], Ah, Qh, integ, PI)
        any(isnan.(x[i].μ)) && error("NaN mean after smoothing")
        any(isnan.(x[i].Σ)) && error("NaN cov after smoothing")
        x[i] = PI * x[i]
    end
end


function smooth!(x_curr, x_next, Ah, Qh, integ, PI=I)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    # PDMat(Symmetric(x_curr.Σ))
    # PDMat(Symmetric(x_next.Σ))

    @unpack d, q = integ.cache
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
    GDG = G * cov_diff * G'
    # GDG = x_curr.Σ * Ah' * (P_p \ (
    #     x_curr.Σ * Ah' * (P_p \ cov_diff')
    # )')
    x_tmp.Σ .= x_curr.Σ .+ GDG
    # approx_diff!(x_tmp.Σ, x_curr.Σ, -GDG)
    # copy!(x_curr.Σ, x_tmp.Σ)
    # Joseph-Form:
    # P = copy(x_curr.Σ)
    # C_tilde = Ah
    # K_tilde = P * Ah' * P_p_inv
    # P_s = ((I - K_tilde*C_tilde) * P * (I - K_tilde*C_tilde)'
    #        + K_tilde * Qh * K_tilde' + G * x_next.Σ * G')
    # P_s = Symmetric(
    #     X_A_Xt(PDMat(Symmetric(P)), (I - K_tilde*C_tilde))
    #     + X_A_Xt(PDMat(Symmetric(Qh)), K_tilde)
    #     + X_A_Xt(PDMat(Symmetric(x_next.Σ)), G)
    # )
    # x_curr.Σ .= P_s

    # fix_negative_variances(x_curr, integ.opts.abstol, integ.opts.reltol)
    assert_nonnegative_diagonal(x_curr.Σ)
    # PDMat(Symmetric(x_curr.Σ))

    return nothing
end
