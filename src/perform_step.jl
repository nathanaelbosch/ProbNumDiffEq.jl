# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    if integ.f isa DynamicalODEFunction &&
       !(integ.sol.prob.problem_type isa SecondOrderODEProblem)
        error(
            """
          The given problem is a `DynamicalODEProblem`, but not a `SecondOrderODEProblem`.
          This can not be handled by ProbNumDiffEq.jl right now. Please check if the
          problem can be formulated as a second order ODE. If not, please open a new
          github issue!
          """,
        )
    end

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
    initial_update!(integ, cache)

    # These are necessary since the solution object is not 100% initialized by default
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x_filt, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(
        integ.sol.pu,
        integ.saveiter,
        _gaussian_mul!(cache.pu_tmp, cache.SolProj, cache.x),
    )
    return nothing
end

"""
    perform_step!(integ, cache::GaussianODEFilterCache[, repeat_step=false])

Perform the ODE filter step.

Basically consists of the following steps
- Compute the current transition and diffusion matrices
- Predict mean
- Evaluate the ODE (and Jacobian) at the predicted mean; Build measurement mean `z`
- Compute local diffusion and local error estimate
- If the step is rejected, terminate here; Else continue
- Predict the covariance and build the measurement covariance `S`
- Kalman update step
- (optional) Update the global diffusion MLE

As in OrdinaryDiffEq.jl, this step is not necessarily successful!
For that functionality, use `OrdinaryDiffEq.step!(integ)`.
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
    @. Ah .= PI.diag .* A .* P.diag'
    X_A_Xt!(Qh, Q, PI)

    # Predict the mean
    predict_mean!(x_pred, x, Ah)
    mul!(view(u_pred, :), SolProj, x_pred.μ)

    # Measure
    evaluate_ode!(integ, x_pred, tnew)

    # Estimate diffusion, and (if adaptive) the local error estimate; Stop here if rejected
    cache.local_diffusion = estimate_local_diffusion(cache.diffusionmodel, integ)
    if integ.opts.adaptive
        integ.EEst = compute_scaled_error_estimate!(integ, cache)
        if integ.EEst >= one(integ.EEst)
            return
        end
    end

    # Predict the covariance, using either the local or global diffusion
    extrapolation_diff =
        isdynamic(cache.diffusionmodel) ? cache.local_diffusion : cache.default_diffusion
    predict_cov!(x_pred, x, Ah, Qh, cache.C_DxD, cache.C_2DxD, extrapolation_diff)

    # Compute measurement covariance only now; likelihood computation is currently broken
    compute_measurement_covariance!(cache)
    # cache.log_likelihood = logpdf(cache.measurement, zeros(d))
    # integ.sol.log_likelihood += integ.cache.log_likelihood

    # Update state and save the ODE solution value
    x_filt = update!(integ, x_pred)
    mul!(view(u_filt, :), SolProj, x_filt.μ)
    integ.u .= u_filt

    # Update the global diffusion MLE (if applicable)
    if !isdynamic(cache.diffusionmodel)
        cache.global_diffusion = estimate_global_diffusion(cache.diffusionmodel, integ)
    end

    # Advance the state
    copy!(integ.cache.x, integ.cache.x_filt)

    return nothing
end

"""
    evaluate_ode!(integ, x_pred, t)

Evaluate the ODE vector field and, if using the [`EK1`](@ref), its Jacobian.

In addition, compute the measurement mean (`z`) and the measurement function Jacobian (`H`).
Results are saved into `integ.cache.du`, `integ.cache.ddu`, `integ.cache.measurement.μ`
and `integ.cache.H`.
Jacobians are computed either with the supplied `f.jac`, or via automatic differentiation,
as in OrdinaryDiffEq.jl.

For second-order ODEs and the `EK1FDB` algorithm a specialized implementation is called.
"""
evaluate_ode!(integ, x_pred, t) =
    evaluate_ode!(integ, x_pred, t, Val(integ.f isa DynamicalODEFunction))
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
    f(du, u_pred, p, t)
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

    # If EK1, evaluate the Jacobian and adjust H
    if alg isa EK1

        # Jacobian is computed either with the given jac, or ForwardDiff
        if !isnothing(f.jac)
            f.jac(ddu, u_pred, p, t)
        else
            !isnothing(f.jac)
            @unpack du1, uf, jac_config = integ.cache
            uf.f = OrdinaryDiffEq.nlsolve_f(f, alg)
            uf.t = t
            if !(p isa DiffEqBase.NullParameters)
                uf.p = p
            end
            OrdinaryDiffEq.jacobian!(ddu, uf, u_pred, du1, integ, jac_config)
        end
        integ.destats.njacs += 1

        # _matmul!(H, f.mass_matrix, E1) # This is already the case (see above)
        _matmul!(H, ddu, E0, -1.0, 1.0)
    end

    return nothing
end

function evaluate_ode!(
    integ::OrdinaryDiffEq.ODEIntegrator{<:EK1FDB},
    x_pred,
    t,
    second_order::Val{false},
)
    @unpack f, p, alg = integ
    @unpack d, u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)

    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    (f.mass_matrix != I) && error("EK1FDB does not support mass-matrices right now")

    # Mean
    f(du, u_pred, p, t)
    integ.destats.nf += 1
    # z .= MM*E1*x_pred.μ .- du
    H1, H2 = view(H, 1:d, :), view(H, d+1:2d, :)
    z1, z2 = view(z, 1:d), view(z, d+1:2d)

    H1 .= E1
    _matmul!(z1, H1, x_pred.μ)
    z1 .-= @view du[:]

    # If EK1, evaluate the Jacobian and adjust H
    if !isnothing(f.jac)
        f.jac(ddu, u_pred, p, t)
    else
        ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_pred)
        integ.destats.nf += 1
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
            f(du, u_pred, p, t)
            if !isnothing(f.jac)
                f.jac(ddu, u_pred, p, t)
            else
                ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u_pred)
                # integ.destats.nf += 1
            end
            # integ.destats.njacs += 1
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
    @unpack f, p, alg = integ
    @unpack d, u_pred, du, ddu, measurement, R, H = integ.cache
    @assert iszero(R)
    du2 = du

    @unpack E0, E1, E2 = integ.cache

    z, S = measurement.μ, measurement.Σ

    # Mean
    # _u_pred = E0 * x_pred.μ
    # _du_pred = E1 * x_pred.μ
    f.f1(du2, view(u_pred, 1:d), view(u_pred, d+1:2d), p, t)
    integ.destats.nf += 1
    _matmul!(z, E2, x_pred.μ)
    z .-= @view du2[:]

    # Cov
    if alg isa EK1
        H .= E2

        J = ddu
        ForwardDiff.jacobian!(
            J,
            (du2, du_u) -> f.f1(du2, view(du_u, 1:d), view(du_u, d+1:2d), p, t),
            du2,
            u_pred,
        )
        integ.destats.nf += 1
        integ.destats.njacs += 1
        _matmul!(H, J, integ.cache.SolProj, -1.0, 1.0)
    end

    return measurement
end

compute_measurement_covariance!(cache) =
    X_A_Xt!(cache.measurement.Σ, cache.x_pred.Σ, cache.H)

function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    @unpack K1, K2, x_tmp2, m_tmp, C_DxD = integ.cache
    update!(x_filt, prediction, measurement, H, K1, C_DxD, m_tmp.Σ)
    return x_filt
end

"""
    compute_scaled_error_estimate!(integ, cache)

Compute the scaled, local error estimate `Eest`, that should satisfy `Eest < 1`.
The actual local error is computed with [`estimate_errors!`](@ref).
Then, `DiffEqBase.calculate_residuals!` handles the scaling with adaptive and relative
tolerances, and `integ.opts.internalnorm` provides the norm that should be used to return
only a scalar.
"""
function compute_scaled_error_estimate!(integ, cache)
    @unpack u_pred, err_tmp = cache
    t = integ.t + integ.dt
    err_est_unscaled = estimate_errors!(cache)
    if integ.f isa DynamicalODEFunction # second-order ODE
        DiffEqBase.calculate_residuals!(
            err_tmp,
            integ.dt * err_est_unscaled,
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
            integ.dt * err_est_unscaled,
            integ.u,
            u_pred,
            integ.opts.abstol,
            integ.opts.reltol,
            integ.opts.internalnorm,
            t,
        )
    end
    return integ.opts.internalnorm(err_tmp, t) # scalar
end

"""
    estimate_errors!(cache)

Computes a local error estimate, as
```math
E_i = ( σ_{loc}^2 ⋅ (H Q(h) H^T)_{ii} )^(1/2)
```
To save allocations, the function modifies the given `cache` and writes into
`cache.C_Dxd` during some computations.
"""
function estimate_errors!(cache::GaussianODEFilterCache)
    @unpack local_diffusion, Qh, H, d = cache

    if local_diffusion isa Real && isinf(local_diffusion)
        return Inf
    end

    R = cache.C_Dxd

    if local_diffusion isa Diagonal
        _matmul!(R, Qh.R * sqrt.(local_diffusion), H')
        error_estimate = sqrt.(diag(SRMatrix(R)))
        return view(error_estimate, 1:d)
    elseif local_diffusion isa Number
        _matmul!(R, Qh.R, H')
        # error_estimate = local_diffusion .* diag(L*L')
        error_estimate = diag(SRMatrix(R))
        error_estimate .*= local_diffusion

        # @info "it's small anyways I guess?" error_estimate cache.measurement.μ .^ 2
        # error_estimate .+= cache.measurement.μ .^ 2

        error_estimate .= sqrt.(error_estimate)
        return view(error_estimate, 1:d)
    end
end
