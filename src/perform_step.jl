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
    @info "New perform_step!" t dt

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
            manifold_update!(x_filt, (x) -> integ.alg.manifold(SolProj * PI * x), integ.alg.mprojmaxiters)
        end

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict!(x_pred, x, A, apply_diffusion(Q, integ.cache.diffusion))

        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            error("It's a bit unclear how to properly handle dynamic diffusion and manifold projections")
            manifold_update!(x_pred, (x) -> integ.alg.manifold(SolProj * PI * x), integ.alg.mprojmaxiters)
        end

        copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        # @info "after predict!" integ.alg.manifold(SolProj * PI * x_pred.μ) |> norm
        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            manifold_update!(x_pred, (x) -> integ.alg.manifold(SolProj * PI * x))
        end
        # @info "after manifold_update! 1" integ.alg.manifold(SolProj * PI * x_pred.μ) |> norm
        mul!(u_pred, SolProj, PI*x_pred.μ)
        measure!(integ, x_pred, tnew)
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)

    end

    # Likelihood
    cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)

    # Project onto the manifold
    if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:after, :both)
        @info "after update!" integ.alg.manifold(SolProj * PI * x_filt.μ) |> norm
        manifold_update!(x_filt, (x) -> integ.alg.manifold(SolProj * PI * x), integ.alg.mprojmaxiters, integ.alg.mprojtime==:both)
        @info "after manifold_update! 2" integ.alg.manifold(SolProj * PI * x_filt.μ) |> norm
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
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u, u_filt,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
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


function manifold_update!(x, h, maxiters=1, check=false)
    z_before = h(x.μ)
    if iszero(z_before) || (check && z_before < eps(typeof(z_before)))
        return
    end

    for i in 1:maxiters
        if i > 1
            @warn "Another iteration of the manifold projection!" i
        end
        z = h(x.μ)
        H = ForwardDiff.gradient(h, x.μ)
        @assert H isa AbstractVector

        SL = H'x.Σ.squareroot
        S = SL*SL'
        K = x.Σ * H * inv(S)
        @info "manifold_update!" z S inv(S) SL SL*SL'

        x.μ .= x.μ .+ K * (0 .- z)
        Pnew = X_A_Xt(x.Σ, (I-K*H'))
        copy!(x.Σ, Pnew)

        z_after = h(x.μ)
        # @info "Iteration" i z_before S z_after z_before ≈ z_after
        # @assert abs(z_after) <= abs(z_before)
        # @assert abs(z_after) <= abs(z_before) || S < eps(typeof(S))
        if iszero(z_after) || S < eps(typeof(S)) break end
        z_before = z_after
    end
    # error()
end

function measure!(integ, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack u_pred, du, ddu, Proj, Precond, measurement, R, H = integ.cache
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
        _eval_f_jac!(ddu, linearize_at, p, t, f)
        integ.destats.njacs += 1
        mul!(H, (E1 .- ddu * E0), PI)
    else
        mul!(H, E1, PI)
    end
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return measurement
end

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
    @unpack diffusion, Q, H = integ.cache

    if diffusion isa Real && isinf(diffusion)
        return Inf
    end

    error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), H))))

    return error_estimate
end
