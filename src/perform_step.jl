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

    integ.kshortsize = 1
    resize!(integ.k, integ.kshortsize)
    integ.k[1] = integ.u

    # Update the initial state to the known (given or computed with AD) initial values
    initial_update!(integ, cache, integ.alg.initialization)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(
        integ.sol.pu,
        integ.saveiter,
        mul!(cache.pu_tmp, cache.SolProj, cache.x),
    )
    return nothing
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
function OrdinaryDiffEq.perform_step!(
    integ,
    cache::GaussianODEFilterCache,
    repeat_step=false,
)
    @unpack t, dt = integ
    @unpack d, SolProj = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack x_tmp, x_tmp2 = integ.cache
    @unpack A, Q, Ah, Qh = integ.cache

    make_preconditioners!(cache, dt)
    @unpack P, PI = integ.cache

    tnew = t + dt

    # Build the correct matrices
    @. Ah = PI.diag .* A .* P.diag'
    X_A_Xt!(Qh, Q, PI)

    if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

        # Predict
        predict_mean!(x_pred, x, Ah)
        mul!(view(u_pred, :), SolProj, x_pred.μ)

        # Measure
        evaluate_ode!(integ, x_pred, tnew)

        # Estimate diffusion
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict_cov!(x_pred, x, Ah, Qh, cache.C1, cache.global_diffusion)

        # Compute measurement covariance only now
        compute_measurement_covariance!(cache)

    else
        predict_mean!(x_pred, x, Ah)
        predict_cov!(x_pred, x, Ah, Qh, cache.C1)
        mul!(view(u_pred, :), SolProj, x_pred.μ)
        evaluate_ode!(integ, x_pred, tnew)
        compute_measurement_covariance!(cache)
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
    end

    # Likelihood
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)

    # Estimate error for adaptive steps - can already be done before filtering
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(cache)
        if integ.f isa DynamicalODEFunction # second-order ODE
            DiffEqBase.calculate_residuals!(
                err_tmp,
                dt * err_est_unscaled,
                integ.u[1, :],
                u_pred[1, :],
                integ.opts.abstol,
                integ.opts.reltol,
                integ.opts.internalnorm,
                t,
            )
        else # regular first-order ODE
            DiffEqBase.calculate_residuals!(
                err_tmp,
                dt * err_est_unscaled,
                integ.u,
                u_pred,
                integ.opts.abstol,
                integ.opts.reltol,
                integ.opts.internalnorm,
                t,
            )
        end
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar
    end

    # If the step gets rejected, we don't even need to perform an update!
    reject = integ.opts.adaptive && integ.EEst >= one(integ.EEst)
    if !reject

        # Save into u_filt and integ.u
        mul!(view(u_filt, :), SolProj, x_filt.μ)
        if integ.u isa Number
            integ.u = u_filt[1]
        else
            integ.u .= u_filt
        end

        # Advance the state here
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
end

function evaluate_ode!(
    integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK},
    x_pred,
    t,
    second_order::Val{false},
)
    @unpack f, p, dt, alg = integ
    @unpack u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)

    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1
    # z .= MM*E1*x_pred.μ .- du
    if f.mass_matrix == I
        H .= E1
    elseif f.mass_matrix isa UniformScaling
        H .= f.mass_matrix.λ .* E1
    else
        _matmul!(H, f.mass_matrix, E1)
    end

    _matmul!(z, H, x_pred.μ)
    z .-= du[:]

    # Cov
    if alg isa EK1 || alg isa IEKS
        u_lin =
            (alg isa IEKS && !isnothing(alg.linearize_at)) ? alg.linearize_at(t).μ : u_pred

        # Jacobian is computed either with the given jac, or ForwardDiff
        if !isnothing(f.jac)
            _eval_f_jac!(ddu, u_lin, p, t, f)
        elseif isinplace(f)
            ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_lin)
        else
            ddu .= ForwardDiff.jacobian(u -> f(u, p, t), u_lin)
        end
        integ.destats.njacs += 1

        # _matmul!(H, f.mass_matrix, E1) # This is already the case (see above)
        _matmul!(H, ddu, E0, -1.0, 1.0)
    else
        # _matmul!(H, f.mass_matrix, E1) # This is already the case (see above)
    end

    return nothing
end

function evaluate_ode!(
    integ::OrdinaryDiffEq.ODEIntegrator{<:EK1FDB},
    x_pred,
    t,
    second_order::Val{false},
)
    @unpack f, p, dt, alg = integ
    @unpack d, u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)

    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    (f.mass_matrix != I) && error("EK1FDB does not support mass-matrices right now")

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1
    # z .= MM*E1*x_pred.μ .- du
    H1, H2 = view(H, 1:d, :), view(H, d+1:2d, :)
    z1, z2 = view(z, 1:d), view(z, d+1:2d)

    H1 .= E1
    _matmul!(z1, H1, x_pred.μ)
    z1 .-= du[:]

    # Cov
    u_lin = u_pred
    if !isnothing(f.jac)
        _eval_f_jac!(ddu, u_lin, p, t, f)
    elseif isinplace(f)
        ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_lin)
    else
        ddu .= ForwardDiff.jacobian(u -> f(u, p, t), u_lin)
    end
    integ.destats.njacs += 1
    _matmul!(H1, ddu, E0, -1.0, 1.0)

    z2 .= (E2 * x_pred.μ .- ddu * du)
    if integ.alg.jac_quality == 1
        # EK0-type approach
        H2 .= E2
    elseif integ.alg.jac_quality == 2
        H2 .= E2 - ddu * ddu * E0
    elseif integ.alg.jac_quality == 3
        _z2(m) = begin
            u_pred = E0 * m
            du = zeros(eltype(m), d)
            ddu = zeros(eltype(m), d, d)
            _eval_f!(du, u_pred, p, t, f)
            if !isnothing(f.jac)
                _eval_f_jac!(ddu, u_pred, p, t, f)
            elseif isinplace(f)
                ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_pred)
            else
                ddu .= ForwardDiff.jacobian(u -> f(u, p, t), u_pred)
            end
            return (E2 * m .- ddu * du)
        end
        H2 .= ForwardDiff.jacobian(_z2, x_pred.μ)
    else
        error("EK1FDB's `jac_quality` has to be in [1,2,3]")
    end
    return nothing
end

function evaluate_ode!(
    integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEK},
    x_pred,
    t,
    second_order::Val{true},
)
    @unpack f, p, dt, alg = integ
    @unpack d, u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)
    du2 = du

    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    # _u_pred = E0 * x_pred.μ
    # _du_pred = E1 * x_pred.μ
    if isinplace(f)
        f.f1(du2, view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    else
        du2 .= f.f1(view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    end
    integ.destats.nf += 1
    z .= E2 * x_pred.μ .- du2[:]

    # Cov
    if alg isa EK1
        (alg isa IEKS) && error("IEKS is currently not supported for SecondOrderODEProbems")

        if isinplace(f)
            J0 = copy(ddu)
            ForwardDiff.jacobian!(
                J0,
                (du2, u) -> f.f1(du2, view(u_pred, 1:d), u, p, t),
                du2,
                u_pred[d+1:2d],
            )

            J1 = copy(ddu)
            ForwardDiff.jacobian!(
                J1,
                (du2, du) -> f.f1(du2, du, view(u_pred, d+1:2d), p, t),
                du2,
                u_pred[1:d],
            )

            integ.destats.njacs += 2

            H .= E2 .- J0 * E0 .- J1 * E1
        else
            J0 = ForwardDiff.jacobian(
                (u) -> f.f1(view(u_pred, 1:d), u, p, t),
                u_pred[d+1:2d],
            )
            J1 = ForwardDiff.jacobian(
                (du) -> f.f1(du, view(u_pred, d+1:2d), p, t),
                u_pred[1:d],
            )
            integ.destats.njacs += 2
            H .= E2 .- J0 * E0 .- J1 * E1
        end
    else
        # H .= E2 # This is already the case!
    end

    return measurement
end
evaluate_ode!(integ, x_pred, t) =
    evaluate_ode!(integ, x_pred, t, Val(integ.f isa DynamicalODEFunction))

# The following functions are just there to handle both IIP and OOP easily
_eval_f!(du, u, p, t, f::AbstractODEFunction{true}) = f(du, u, p, t)
_eval_f!(du, u, p, t, f::AbstractODEFunction{false}) = (du .= f(u, p, t))
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{true}) = f.jac(ddu, u, p, t)
_eval_f_jac!(ddu, u, p, t, f::AbstractODEFunction{false}) = (ddu .= f.jac(u, p, t))

compute_measurement_covariance!(cache) =
    X_A_Xt!(cache.measurement.Σ, cache.x_pred.Σ, cache.H)

function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    @unpack K1, K2, x_tmp2, m_tmp = integ.cache
    update!(x_filt, prediction, measurement, H, K1, x_tmp2.Σ.mat, m_tmp)
    return x_filt
end

function smooth_all!(integ)
    integ.sol.x_smooth = copy(integ.sol.x_filt)

    @unpack A, Q = integ.cache
    @unpack x_smooth, t, diffusions = integ.sol
    @unpack x_tmp, x_tmp2 = integ.cache
    x = x_smooth

    for i in length(x)-1:-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x[i], x[i+1])
            continue
        end

        make_preconditioners!(integ.cache, dt)
        P, PI = integ.cache.P, integ.cache.PI

        mul!(x_tmp, P, x[i])
        mul!(x_tmp2, P, x[i+1])
        smooth!(x_tmp, x_tmp2, A, Q, integ, diffusions[i])
        mul!(x[i], PI, x_tmp)
    end
end

function estimate_errors(cache::GaussianODEFilterCache)
    @unpack local_diffusion, Qh, H, d = cache

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    L = cache.m_tmp.Σ.squareroot

    if local_diffusion isa Diagonal
        _matmul!(L, H, sqrt.(local_diffusion) * Qh.squareroot)
        error_estimate = sqrt.(diag(L * L'))
        return view(error_estimate, 1:d)

    elseif local_diffusion isa Number
        _matmul!(L, H, Qh.squareroot)
        # error_estimate = local_diffusion .* diag(L*L')
        @tullio error_estimate[i] := L[i, j] * L[i, j]
        error_estimate .*= local_diffusion

        # @info "it's small anyways I guess?" error_estimate cache.measurement.μ .^ 2
        # error_estimate .+= cache.measurement.μ .^ 2

        error_estimate .= sqrt.(error_estimate)
        return view(error_estimate, 1:d)
    end
end
