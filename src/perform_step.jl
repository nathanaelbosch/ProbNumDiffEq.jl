# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    if integ.opts.dense && !integ.alg.smooth
        error("To use `dense=true` you need to set `smooth=true`!")
    elseif !integ.opts.dense && integ.alg.smooth
        @warn "If you set dense=false for efficiency, you might also want to set smooth=false."
    end
    if !integ.opts.save_everystep && integ.alg.smooth
        error("If you do not save all values, you do not need to smooth!")
    end
    @assert integ.saveiter == 1

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter,
                                   mul!(cache.pu_tmp, cache.SolProj, cache.x))
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
    @unpack d, SolProj = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack x_tmp, x_tmp2 = integ.cache
    @unpack A, Q = integ.cache

    make_preconditioners!(integ, dt)
    @unpack P, PI = integ.cache

    tnew = t + dt

    # Coordinate change / preconditioning
    # x = P * x
    x = mul!(x_tmp, P, x)

    if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

        # Predict
        predict_mean!(x_pred, x, A)
        @. x_tmp2.μ = PI.diag * x_pred.μ
        _matmul!(view(u_pred, :), SolProj, x_tmp2.μ)

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict_cov!(x_pred, x, A, Q, cache.C1, cache.global_diffusion)
        X_A_Xt!(integ.cache.measurement.Σ, x_pred.Σ, integ.cache.H)

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        @. x_tmp2.μ = PI.diag * x_pred.μ
        _matmul!(u_pred, SolProj, x_tmp2.μ)
        measure!(integ, x_pred, tnew)
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
    end

    # Likelihood
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)

    # Undo the coordinate change / preconditioning
    mul!(integ.cache.x, PI, x)
    mul!(integ.cache.x_pred, PI, x_pred)
    mul!(integ.cache.x_filt, PI, x_filt)

    _matmul!(view(u_filt, :), SolProj, x_filt.μ)

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
                integ.u isa Number ? integ.u : integ.u[:], u_filt,
                integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        end
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar
    end

    if integ.u isa Number
        integ.u = u_filt[1]
    else
        integ.u .= u_filt
    end

    # stuff that would normally be in apply_step!
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end

function measure!(integ, x_pred, t, second_order::Val{false})
    @unpack f, p, dt, alg = integ
    @unpack u_pred, du, ddu, measurement, R, H = integ.cache
    @unpack P, PI = integ.cache
    @assert iszero(R)

    @unpack E0, E1 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1
    # z .= E1*PI*x_pred.μ .- du
    _matmul!(z, E1, mul!(integ.cache.x_tmp2.μ, PI, x_pred.μ))
    z .-= du[:]

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
        _matmul!(H, (E1 .- ddu * E0), PI)
    else
        _matmul!(H, E1, PI)
    end
    X_A_Xt!(S, x_pred.Σ, H)

    return measurement
end

function measure!(integ, x_pred, t, second_order::Val{true})
    @unpack f, p, dt, alg = integ
    @unpack d, u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)
    du2 = du

    @unpack P, PI = integ.cache
    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    # _u_pred = E0 * PI * x_pred.μ
    # _du_pred = E1 * PI * x_pred.μ
    if isinplace(f)
        f.f1(du2, view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    else
        du2 .= f.f1(view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    end
    integ.destats.nf += 1
    z .= E2*PI*x_pred.μ .- du2

    # Cov
    if alg isa EK1
        @assert !(alg isa IEKS)

        if isinplace(f)
            J0 = copy(ddu)
            ForwardDiff.jacobian!(J0, (du2, u) -> f.f1(du2, view(u_pred, 1:d), u, p, t), du2,
                                  u_pred[d+1:2d])

            J1 = copy(ddu)
            ForwardDiff.jacobian!(J1, (du2, du) -> f.f1(du2, du, view(u_pred, d+1:2d),
                                                        p, t), du2,
                                  u_pred[1:d])

            integ.destats.njacs += 2

            _matmul!(H, (E2 .- J0 * E0 .- J1 * E1), PI)
        else
            J0 = ForwardDiff.jacobian((u) -> f.f1(view(u_pred, 1:d), u, p, t), u_pred[d+1:2d])
            J1 = ForwardDiff.jacobian((du) -> f.f1(du, view(u_pred, d+1:2d), p, t), u_pred[1:d])
            integ.destats.njacs += 2
            _matmul!(H, (E2 .- J0 * E0 .- J1 * E1), PI)
        end
    else
        _matmul!(H, E2, PI)
    end

    X_A_Xt!(S, x_pred.Σ, H)

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
    @unpack K1, K2, x_tmp2 = integ.cache
    update!(x_filt, prediction, measurement, H, R, K1, K2, x_tmp2.Σ.mat)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack local_diffusion, Q, H = cache

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    L = cache.m_tmp.Σ.squareroot

    if local_diffusion isa Diagonal

        mul!(L, H, sqrt.(local_diffusion) * Q.squareroot)
        error_estimate = sqrt.(diag(L*L'))
        return error_estimate

    elseif local_diffusion isa Number

        mul!(L, H, Q.squareroot)
        # error_estimate = local_diffusion .* diag(L*L')
        @tullio error_estimate[i] := L[i,j]*L[i,j]
        error_estimate .*= local_diffusion
        error_estimate .= sqrt.(error_estimate)
        return error_estimate

    end
end
