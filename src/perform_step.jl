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
        # predict_mean!(x_pred, x, A, Q)
        predict!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)

        # Proj
        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            error("It's a bit unclear how to properly handle dynamic diffusion and manifold projections")
            update!(x_filt, x_filt, (x) -> integ.alg.manifold(SolProj * PI * x))
        end

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict!(x_pred, x, A, apply_diffusion(Q, integ.cache.global_diffusion))

        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            error("It's a bit unclear how to properly handle dynamic diffusion and manifold projections")
            update!(x_pred, x_pred, (x) -> integ.alg.manifold(SolProj * PI * x))
        end

        copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        # @info "after predict!" integ.alg.manifold(SolProj * PI * x_pred.μ) |> norm
        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            x_pred_new = iekf_update(x_pred, (x) -> integ.alg.manifold(SolProj * PI * x),
                                     maxiters=integ.alg.mprojiekf ? 100 : 1)
            copy!(x_pred, x_pred_new)
        end
        # @info "after manifold_update! 1" integ.alg.manifold(SolProj * PI * x_pred.μ) |> norm
        mul!(u_pred, SolProj, PI*x_pred.μ)
        measure!(integ, x_pred, tnew)
        cache.local_diffusion, cache.global_diffusion =
            estimate_diffusion(cache.diffusionmodel, integ)
    end

    # Likelihood
    # cache.log_likelihood = logpdf(cache.measurement, zero(cache.measurement.μ))

    # Update
    x_filt = update!(integ, x_pred)

    # Project onto the manifold
    if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:after, :both)
        x_filt_new = iekf_update(x_filt, (x) -> integ.alg.manifold(SolProj * PI * x);
                                 maxiters=integ.alg.mprojiekf ? 100 : 1)
        copy!(x_filt, x_filt_new)
    end

    # Save
    mul!(u_filt, SolProj, PI*x_filt.μ)

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        if integ.f isa DynamicalODEFunction # second order solver
            DiffEqBase.calculate_residuals!(
                err_tmp, dt * err_est_unscaled,
                integ.u[2, :], u_filt[2, :],
                integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        else
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
    # (t > 0) && error()
end

function measure!(integ, x_pred, t, second_order::Val{false})
    @unpack f, p, dt, alg = integ
    @unpack u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
    P = Precond(dt); PI = inv(P)
    E0, E1 = Proj(0), Proj(1)

    z, S = measurement.μ, measurement.Σ

    # Mean
    _eval_f!(du, u_pred, p, t, f)
    integ.destats.nf += 1

    # Cov
    if alg isa EK1 || alg isa IEKS
        linearize_at = (alg isa IEKS && !isnothing(alg.linearize_at)) ?
            alg.linearize_at(t).μ : u_pred
        _eval_f_jac!(ddu, linearize_at, p, t, f)
        integ.destats.njacs += 1
        if alg.fdb_improved == 0
            z .= E1*PI*x_pred.μ .- du
            mul!(H, (E1 .- ddu * E0), PI)
        else
            E2 = Proj(2)
            z2_ = z2(x_pred.μ, integ, t)
            # @info "?" z2_ E2*PI*x_pred.μ .- ddu * du
            # @assert z2_ ≈ E2*PI*x_pred.μ .- ddu * du
            z[1:d] .= E1*PI*x_pred.μ .- du
            z[d+1:end] .= z2_
            # @info "measure!" du ddu

            H[1:d, :] .= (E1 .- ddu * E0) * PI
            if alg.fdb_improved==1
                H[d+1:end, :] .= E2*PI
            elseif alg.fdb_improved==2
                H[d+1:end, :] .= E2 * PI .- ddu * ddu * E0 * PI
            elseif alg.fdb_improved==3
                Jz2 = ForwardDiff.jacobian(Y -> z2(Y, integ, t), x_pred.μ)
                H[d+1:end, :] .= Jz2
            end
        end
    else
        z .= E1*PI*x_pred.μ .- du
        mul!(H, E1, PI)
    end
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return measurement
end

function z2(Y, integ, t)
    @unpack f, p, dt, alg = integ
    @unpack d, Proj, Precond = integ.cache
    P = Precond(dt); PI = inv(P)
    E0, E1, E2 = Proj(0), Proj(1), Proj(2)

    du = copy(E1*Y)
    _eval_f!(du, E0*PI*Y, p, t, f)
    ddu = zeros(eltype(Y), d, d)
    _eval_f_jac!(ddu, E0*PI*Y, p, t, f)

    z = E2*PI*Y .- ddu * du

    return z
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
    f.f1(du2, u_pred[1:d], u_pred[d+1:end], p, t)
    integ.destats.nf += 1
    z .= E2*PI*x_pred.μ .- du2

    # Cov
    if alg isa EK1
        @assert !(alg isa IEKS)
        ForwardDiff.jacobian!(ddu, (du2, u) -> f.f1(du2, u_pred[1:d], u, p, t), du2, u_pred[d+1:end])
        Ju = ddu
        integ.destats.njacs += 1
        mul!(H, (E2 .- Ju * E0), PI)
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
    @unpack local_diffusion, Q, H, Precond, Proj, d = integ.cache
    PI = inv(Precond(integ.dt))
    E0 = Proj(0)

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    _error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), H))))

    error_estimate = _error_estimate[1:d]
    # error_estimate = integ.opts.internalnorm(_error_estimate, integ.t)

    # error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), E0*PI))))

    # _Q = apply_diffusion(Q, diffusion)
    # p_filt_local = X_A_Xt(_Q, I - H'H*_Q)
    # error_estimate = sqrt.(diag(Matrix(X_A_Xt(p_filt_local, E0*PI))))

    return error_estimate
end
