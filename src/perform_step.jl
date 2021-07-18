# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
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
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache

    tnew = t + dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = inv(P)
    x = P * x

    if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

        # Predict
        predict_mean!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict_cov!(x_pred, x, A, apply_diffusion(Q, cache.global_diffusion))
        copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)
        measure!(integ, x_pred, tnew)
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
    end

    # Likelihood
    cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)
    mul!(u_filt, SolProj, PI*x_filt.μ)

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        if integ.f isa DynamicalODEFunction # second-order ODE
            DiffEqBase.calculate_residuals!(
                err_tmp, dt * err_est_unscaled,
                integ.u[1, :], u_filt[1, :],
                integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        else # regular first-order ODE
            DiffEqBase.calculate_residuals!(
                err_tmp, dt * err_est_unscaled,
                integ.u, u_filt,
                integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        end
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar
    end

    integ.u .= u_filt

    # stuff that would normally be in apply_step!
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end

function measure!(integ, x_pred, t, second_order::Val{false})
    @unpack f, p, dt, alg = integ
    @unpack u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
    @assert iszero(R)

    PI = inv(Precond(dt))
    E0, E1 = Proj(0), Proj(1)

    z, S = measurement.μ, measurement.Σ

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1
    z .= E1*PI*x_pred.μ .- du

    # Cov
    if alg isa EK1 || alg isa IEKS
        linearize_at = (alg isa IEKS && !isnothing(alg.linearize_at)) ?
            alg.linearize_at(t).μ : u_pred

        # Jacobian is now computed either with the given jac, or ForwardDiff
        if !isnothing(f.jac)
            _eval_f_jac!(ddu, linearize_at, p, t, f)
        elseif isinplace(f)
            ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_pred)
        else
            ddu .= ForwardDiff.jacobian(u -> f(u, p, t), u_pred)
        end

        integ.destats.njacs += 1
        mul!(H, (E1 .- ddu * E0), PI)
    else
        mul!(H, E1, PI)
    end
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return measurement
end

function measure!(integ, x_pred, t, second_order::Val{true})
    @unpack f, p, dt, alg = integ
    @unpack d, u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
    @assert iszero(R)
    du2 = du

    PI = inv(Precond(dt))
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)

    z, S = measurement.μ, measurement.Σ

    # Mean
    # _u_pred = E0 * PI * x_pred.μ
    # _du_pred = E1 * PI * x_pred.μ
    @assert isinplace(f) "Currently the code only supports IIP `SecondOrderProblem`s"
    f.f1(du2, view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    integ.destats.nf += 1
    z .= E2*PI*x_pred.μ .- du2

    # Cov
    if alg isa EK1
        @assert !(alg isa IEKS)

        J0 = copy(ddu)
        ForwardDiff.jacobian!(J0, (du2, u) -> f.f1(du2, view(u_pred, 1:d), u, p, t), du2,
                              u_pred[d+1:2d])

        J1 = copy(ddu)
        ForwardDiff.jacobian!(J1, (du2, du) -> f.f1(du2, du, view(u_pred, d+1:2d),
                                                   p, t), du2,
                              u_pred[1:d])

        integ.destats.njacs += 1
        mul!(H, (E2 .- J0 * E0 .- J1 * E1), PI)
    else
        mul!(H, E2, PI)
    end

    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return measurement
end
measure!(integ, x_pred, t) = measure!(
    integ, x_pred, t, Val(integ.f isa DynamicalODEFunction))

# The following functions are just there to handle both IIP and OOP easily
_eval_f!(du, u, p, t, f::AbstractODEFunction{true}) = f(du, u, p, t)
_eval_f!(du, u, p, t, f::AbstractODEFunction{false}) = (du .= f(u, p, t))
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{true}) = f.jac(ddu, u, p, t)
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{false}) = (ddu .= f.jac(u, p, t))

function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    update!(x_filt, prediction, measurement, H, R)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack local_diffusion, Q, H = cache

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, local_diffusion), H))))

    return error_estimate
end
