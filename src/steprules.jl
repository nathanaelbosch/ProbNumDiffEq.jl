""" Constant steps """
function constant_steprule()
    function steprule(integ, proposal, proposals)
        accept, h_new = true, integ.dt
        return accept, h_new
    end
end


""" Accept/reject and increase/decrease depending on the p-value of the error

DISCONTINUED FOR NOW!
"""
function pvalue_steprule(tol)
    counter = 0
    function steprule(integ, proposal, proposals)
        @unpack d, dt = integ
        error("p-value steprule currently broken")
        # current_h, current_error, previous_sigma, d, argv...)
        pval = cdf(Chisq(d), current_error / previous_sigma)
        h_new = current_h * 0.95
        if (pval < 1-tol)
            counter = 0
            accept = true
            h_new = h_new * 1.5
        elseif (counter > 10)
            accept = true
        else
            counter += 1
            accept = false
            h_new = h_new * 0.5
        end
        return accept, h_new
    end
    return steprule
end


"""Limit the function error to provided tolerances

This is a /local/ approximation; At each step we assume, that the
previous step had correct results"""
function classic_steprule(abstol, reltol, scale=1; ρ=0.95)
    function steprule(integ, proposal, proposals)
        @unpack dm, d, q, dt = integ
        @unpack measurement, σ², prediction = proposal

        if σ² == 1
            σ² = static_sigma_estimation(
                integ.sigma_estimator, integ,
                [proposals; (proposal..., accept=true, t=t, dt=dt)])
        end

        # Predict step, assuming a correct current estimate (P=0)
        # The prediction step therefore provides P_p=Q
        # *NOTE*: This does not seem to make that much difference!
        # @unpack H, R = proposal
        # P_loc =
        # P_loc = σ² * dm.Q(dt)
        # S = H * P_loc * H' + R
        # K = P_loc * H' * inv(S)
        # P_loc = P_loc - K * S * K'
        # f_cov = P_loc[1:d, 1:d]

        f_cov = σ² * dm.Q(dt)[1:d, 1:d]
        # @assert isdiag(f_cov)
        f_err = sqrt.(diag(f_cov)) * scale
        tol = (abstol .+ (reltol * abs.(prediction.μ[1:d])))
        f_err_scaled = norm(f_err ./ tol)
        # f_err_scaled /= current_h  # Error per unit, not per step

        accept = f_err_scaled <= 1
        if !accept
            # @info "Rejected h=$current_h with scaled error e=$f_err_scaled !"
        end

        h_proposal = dt * ρ * (1/f_err_scaled)^(1/(2q+1))
        h_new = min(max(h_proposal, dt*0.1), dt*5)

        return accept, h_new
    end
    return steprule
end



"""Limit the measurement error to provided tolerances

This is a /local/ approximation; At each step we assume, that the
previous step had correct results"""
function measurement_error_steprule(;ρ=0.95, abstol=1e-6, reltol=1e-3)
    function steprule(integ, proposal, proposals)
        @unpack dm, d, q, dt, t = integ
        @unpack measurement, σ², prediction = proposal
        # S = previous_sigma .* measurement.Σ

        σ² = static_sigma_estimation(
            integ.sigma_estimator, integ,
            [proposals; (proposal..., accept=true, t=t, dt=dt)])

        S = σ² .* measurement.Σ
        # @assert isdiag(S)
        z_err = sqrt.(diag(S))
        tol = (abstol .+ (reltol * abs.(prediction.μ[1:d])))

        z_err_scaled = maximum(z_err ./ tol)

        accept = z_err_scaled <= 1
        if !accept
            # @info "Rejected h=$current_h with scaled error e=$z_err_scaled !"
        end

        h_proposal = dt * ρ * (1/z_err_scaled)^(1/(2q-1))
        h_new = min(max(h_proposal, dt*0.1), dt*5)

        return accept, h_new
    end
    return steprule
end


"""This is basically the steprule with I discussed with Filip"""
function measurement_scaling_steprule(abstol=1, reltol=0; ρ=0.8, hmin=1e-5)
    function steprule(integ, proposal, proposals)
        @unpack dm, d, q, dt = integ
        @unpack measurement, σ², prediction = proposal

        σ² = static_sigma_estimation(
            integ.sigma_estimator, integ,
            [proposals; (proposal..., accept=true, t=t, dt=dt)])
        residual = measurement.μ' * inv(measurement.Σ) * measurement.μ / σ²
        @show residual
        @show σ²

        accept = residual <= 1
        # !accept && @info "Rejected h=$current_h with scaled error e=$calibrated_error !"

        h_exp = 1
        h_proposal = ρ * dt * (1/residual)^(1/h_exp)
        # @show h_proposal, h_proposal/current_h, accept
        h_new = min(max(h_proposal, dt*0.5), dt*2)

        if h_new <= hmin
            error("Step size too small")
        end

        return accept, h_new
    end
    return steprule
end


"""Implementation of the steprule from Michael Schober

It is not 100% faithful to the paper. For example, I do not use the specified
weights, and I just norm over all dimensions instead of considering all of them
separately.
"""
function schober16_steprule(; ρ=0.95, abstol=1e-6, reltol=1e-3, hmin=1e-6)
    function steprule(integ, proposal, proposals)
        @unpack dm, mm, q, d, dt, t = integ
        @unpack t, prediction, measurement = proposal
        h = dt

        v = measurement.μ
        Q = dm.Q(dt)
        H = mm.H(prediction.μ, t)
        # σ² = v' * inv(H*Q*H') * v / length(v)
        @assert typeof(integ.sigma_estimator) == Schober16Sigma
        σ² = dynamic_sigma_estimation(integ.sigma_estimator; H=H, Q=Q, v=v)

        w = ones(d)
        D = sqrt.(diag(H * σ²*dm.Q(h) * H')) .* w

        ϵ = (abstol .+ (reltol * abs.(prediction.μ[1:d])))

        D = maximum(D ./ ϵ)

        # S = h
        # ϵ_ = ϵ * h / S
        # accept = D <= ϵ_
        accept = D <= 1
        h_proposal = h * ρ * (1 / D)^(1/(2q-1))
        h_new = min(max(h_proposal, dt*0.1), dt*5)

        if h_new <= hmin
            error("Step size too small")
        end

        return accept, h_new
    end
    return steprule
end
