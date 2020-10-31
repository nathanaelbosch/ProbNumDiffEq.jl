# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    @assert integ.saveiter == 1
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, copy(integ.cache.x))
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, integ.cache.E0*integ.cache.x)
end

"""Perform a step

Not necessarily successful! For that, see `step!(integ)`.

Basically consists of the following steps
- Coordinate change / Predonditioning
- Prediction step
- Measurement: Evaluate f and Jf; Build z, S, H
- Calibration; Adjust prediction / measurement covs if the diffusion model "dynamic"
- Update step
- Error estimation
- Undo the coordinate change / Predonditioning
"""
function OrdinaryDiffEq.perform_step!(integ, cache::GaussianODEFilterCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, E0, Precond, InvPrecond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A!, Q!, Ah, Qh = integ.cache

    tnew = t + dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = InvPrecond(dt)
    x = P * x

    # Dynamics for this step
    A!(Ah, dt)
    Q!(Qh, dt)

    # Predict
    predict!(x_pred, x, Ah, Qh)
    mul!(u_pred, E0, PI*x_pred.μ)

    # Measure
    measure!(integ, x_pred, tnew)

    # Estimate diffusion
    diffmat = estimate_diffusion(cache.diffusionmodel, integ)
    integ.cache.diffmat = diffmat
    if isdynamic(cache.diffusionmodel) # Adjust prediction and measurement
        predict!(x_pred, x, Ah, diffmat .* Qh)
        integ.cache.measurement.Σ .+=
            integ.cache.H * ((diffmat .- 1) .* Qh) * integ.cache.H'
    end

    # Likelihood
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))
    cache.log_likelihood = 0
    # TODO: The correct one fails in tests due to cholesky stuff

    # Update
    x_filt = update!(integ, x_pred)
    mul!(u_filt, E0, PI*x_filt.μ)
    integ.u .= u_filt

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u, u_filt,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar
    end

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)
end


function h!(integ, x_pred, t)
    @unpack f, p, dt = integ
    @unpack du, E0, E1, InvPrecond, measurement = integ.cache
    PI = InvPrecond(dt)
    z = measurement.μ

    u_pred = E0*PI*x_pred.μ
    IIP = isinplace(integ.f)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    z .= E1*PI*x_pred.μ .- du

    return z
end

function H!(integ, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack ddu, E0, E1, InvPrecond, H = integ.cache
    PI = InvPrecond(dt)

    if alg isa EKF1 || alg isa IEKS
        if alg isa IEKS && !isnothing(alg.linearize_at)
            u_pred = alg.linearize_at(t).μ
        else
            u_pred = E0*PI*x_pred.μ
        end

        if isinplace(integ.f)
            f.jac(ddu, u_pred, p, t)
        else
            ddu .= f.jac(u_pred, p, t)
            # WIP: Handle Jacobians as OrdinaryDiffEq.jl does
            # J = OrdinaryDiffEq.jacobian((u)-> f(u, p, t), u_pred, integ)
            # @assert J ≈ ddu
        end
        integ.destats.njacs += 1
    end

    H .= (E1 .- ddu * E0) * PI  # For ekf0 we have ddu==0
    return H
end


function measure!(integ, x_pred, t)
    @unpack R = integ.cache
    @unpack u_pred, measurement, H = integ.cache

    z, S = measurement.μ, measurement.Σ
    z .= h!(integ, x_pred, t)
    H .= H!(integ, x_pred, t)
    # R .= Diagonal(eps.(z))
    @assert iszero(R)
    copy!(S, X_A_Xt(x_pred.Σ, H))

    return nothing
end


function update!(integ, prediction)

    @unpack dt = integ
    @unpack R, q = integ.cache
    @unpack measurement, H, K, x_filt = integ.cache

    z, S = measurement.μ, measurement.Σ

    m_p, P_p = prediction.μ, prediction.Σ

    S_inv = inv(S)
    K .= P_p * H' * S_inv

    x_filt.μ .= m_p .+ K * (0 .- z)

    # Joseph Form
    out_cov = X_A_Xt(P_p, (I-K*H)) # + X_A_Xt(R, K)
    @assert iszero(R)
    copy!(x_filt.Σ, out_cov)

    assert_nonnegative_diagonal(x_filt.Σ)

    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack dt = integ
    @unpack InvPrecond = integ.cache
    @unpack diffmat, Qh, H = integ.cache

    if diffmat isa Real && isinf(diffmat)
        return Inf
    end

    error_estimate = sqrt.(diag(H * (diffmat .* Qh) * H'))

    return error_estimate
end
