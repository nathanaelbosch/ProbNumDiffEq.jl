""" Constant steps """
function constant_steprule()
    function steprule(integ)
        accept, h_new = true, integ.dt
        return accept, h_new
    end
end

function constant_stepsize_controller!(integrator)
    return one(integrator.dt)
end

function standard_stepsize_controller!(integrator)
end

function PI_stepsize_controller!(integrator)
end


"""Limit the function error to provided tolerances

This is a /local/ approximation; At each step we assume, that the
previous step had correct results"""
function classic_steprule(abstol, reltol, scale=1; ρ=0.95)
    function steprule(integ)
        @unpack dt = integ
        @unpack d, q, E0 = integ.constants
        @unpack σ_sq, Qh, x_pred, u_pred = integ.cache

        if σ_sq == 1
            σ_sq = static_sigma_estimation(
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

        f_cov = σ_sq * E0 * Qh * E0'
        # @assert isdiag(f_cov)
        f_err = sqrt.(diag(f_cov)) * scale
        tol = (abstol .+ (reltol * abs.(u_pred)))
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


"""Implementation of the steprule from Michael Schober

It is not 100% faithful to the paper. For example, I do not use the specified
weights, and I just norm over all dimensions instead of considering all of them
separately.
"""
function schober16_steprule(; ρ=0.95, abstol=1e-3, reltol=1e-2, hmin=1e-6)
    function steprule(integ)
        @unpack dt = integ
        @unpack d, q, E0 = integ.constants
        @unpack σ_sq, Qh, x_pred, h, H = integ.cache
        u_pred = E0 * x_pred.μ

        h = dt

        v = h
        @assert typeof(integ.sigma_estimator) == Schober16Sigma
        # σ_sq = dynamic_sigma_estimation(integ.sigma_estimator; H=H, Q=Q, v=v)

        w = ones(d)
        D = sqrt.(diag(H * σ_sq*Qh * H')) .* w

        ϵ = (abstol .+ (reltol * abs.(u_pred)))

        D = maximum(D ./ ϵ)

        # S = h
        # ϵ_ = ϵ * h / S
        # accept = D <= ϵ_
        accept = D <= 1
        h_proposal = h * ρ * (1 / D)^(1/(q+1))
        h_new = min(max(h_proposal, dt*0.1), dt*5)

        if h_new <= hmin
            error("Step size too small")
        end

        return accept, h_new
    end
    return steprule
end



abstract type StepController end
struct ConstantSteps <: StepController end
function propose_step(::ConstantSteps, integ)
    return integ.dt
end


struct StandardSteps <: StepController end
function propose_step(::StandardSteps, integ)

    @unpack dt, EEst = integ
    @unpack q = integ.constants
    @unpack gamma, qmin, qmax = integ.opts

    if iszero(EEst)
        q = inv(qmax)
    else
        localconvrate = q+1
        qtmp = DiffEqBase.fastpow(EEst, 1/localconvrate) / gamma
        @fastmath q = DiffEqBase.value(max(inv(qmax), min(inv(qmin), qtmp)))
    end
    dt /= q
    return dt
end

struct PISteps <: StepController end
function propose_step(::PISteps, integ)
    # PI-controller
    @unpack dt, EEst, qold, q11 = integ
    @unpack beta1, beta2, qmin, qmax = integ.opts

    if iszero(EEst)
        q = inv(qmax)
    else
        q11 = DiffEqBase.value(DiffEqBase.fastpow(EEst, beta1))
        q = q11 / DiffEqBase.fastpow(qold, beta2) / gamma
        integrator.q11 = q11
        @fastmath q = DiffEqBase.value(max(inv(qmax),min(inv(qmin), q)))
    end
    dt /= q
    return dt
end
