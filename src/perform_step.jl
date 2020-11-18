# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    @assert integ.saveiter == 1
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, copy(cache.x))
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
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
    @unpack d, Proj, SolProj, Precond, InvPrecond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache

    tnew = t + dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = InvPrecond(dt)
    x = P * x

    # Predict
    predict!(x_pred, x, A, Q*dt)
    mul!(u_pred, SolProj, PI*x_pred.μ)

    # Measure
    measure!(integ, x_pred, tnew)

    # Estimate diffusion
    diffmat = estimate_diffusion(cache.diffusionmodel, integ)
    integ.cache.diffmat = diffmat
    if isdynamic(cache.diffusionmodel) # Adjust prediction and measurement
        predict!(x_pred, x, A, apply_diffusion(Q*dt, diffmat))
        copy!(integ.cache.measurement.Σ, X_A_Xt(x_pred.Σ, integ.cache.H))
    end

    # Likelihood
    v, S = cache.measurement.μ, cache.measurement.Σ
    cache.log_likelihood = logpdf(Gaussian(v, Symmetric(Matrix(S))), zeros(d))

    # Update
    x_filt = update!(integ, x_pred)
    mul!(u_filt, SolProj, PI*x_filt.μ)
    integ.u .= u_filt

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u[1:d], u_filt[1:d],
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
    @unpack u_pred, du, Proj, InvPrecond, measurement = integ.cache
    PI = InvPrecond(dt)
    z = measurement.μ
    E0, E1 = Proj(0), Proj(1)

    u_pred .= E0*PI*x_pred.μ
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
    @unpack ddu, Proj, InvPrecond, H = integ.cache
    E0, E1 = Proj(0), Proj(1)
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
    @unpack measurement, H, R, x_filt = integ.cache
    update!(x_filt, prediction, measurement, H, R)
    assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack dt = integ
    @unpack InvPrecond = integ.cache
    @unpack diffmat, Q, H = integ.cache

    if diffmat isa Real && isinf(diffmat)
        return Inf
    end

    error_estimate = sqrt.(diag(X_A_Xt(apply_diffusion(Q*dt, diffmat), H)))

    return error_estimate
end
