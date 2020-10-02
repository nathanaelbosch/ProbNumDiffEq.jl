"""Perform a step, but not necessarily successful!

This is the actual interestin part of the algorithm
"""
function perform_step!(integ::ODEFilterIntegrator)
    @unpack t, dt = integ
    @unpack E0, Precond, InvPrecond = integ.constants
    @unpack x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache

    P = Precond(dt)
    PI = InvPrecond(dt)
    integ.cache.x = P * integ.cache.x

    t = t + dt
    integ.t_new = t

    x_pred = predict!(integ)
    mul!(u_pred, E0, PI*x_pred.μ)

    measure!(integ, x_pred, t)

    if isdynamic(integ.sigma_estimator)
        # @info "Before dynamic sigma:" x_pred.Σ
        σ_sq = dynamic_sigma_estimation(integ.sigma_estimator, integ)

        # Adjust prediction and measurement accordingly
        x_pred.Σ .+= (σ_sq .- 1) .* integ.cache.Qh
        integ.cache.measurement.Σ .+= integ.cache.H * ((σ_sq .- 1) .* integ.cache.Qh) * integ.cache.H'

        # @info "After sigma estimation:" t σ_sq x_pred.Σ # (σ_sq > eps(typeof(σ_sq)))
        # assert_good_covariance(x_pred.Σ)

        integ.cache.σ_sq = σ_sq
        # error("Terminate to inspect")
    end

    x_filt = update!(integ, x_pred)
    mul!(u_filt, E0, PI*x_filt.μ)

    if isstatic(integ.sigma_estimator)
        # E.g. estimate the /current/ MLE sigma; Needed for error estimation
        σ_sq = static_sigma_estimation(integ.sigma_estimator, integ)
        integ.cache.σ_sq = σ_sq
    end

    err_est_unscaled = estimate_errors(integ.error_estimator, integ)
    # Scale the error with old u-values and tolerances
    DiffEqBase.calculate_residuals!(
        err_tmp,
        dt * err_est_unscaled, integ.u, u_filt, integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
    err_est_combined = integ.opts.internalnorm(err_tmp, t)  # Norm over the dimensions
    integ.EEst = err_est_combined

    integ.cache.x = PI * integ.cache.x
    integ.cache.x_pred = PI * integ.cache.x_pred
    integ.cache.x_filt = PI * integ.cache.x_filt
end


function predict!(integ::ODEFilterIntegrator)

    @unpack dt = integ
    @unpack A!, Q!, InvPrecond = integ.constants
    @unpack x, Ah, Qh, x_pred = integ.cache
    PI = InvPrecond(dt)

    A!(Ah, dt)
    Q!(Qh, dt)

    mul!(x_pred.μ, Ah, x.μ)
    x_pred.Σ .= Symmetric(Ah * x.Σ * Ah' .+ Qh)

    return x_pred
end


function measure_h!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, f, dt = integ
    @unpack E0, h!, InvPrecond = integ.constants
    @unpack du, h, u_pred = integ.cache
    PI = InvPrecond(dt)

    IIP = isinplace(integ)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    h!(h, du, PI*x_pred.μ)
end

function measure_H!(integ::ODEFilterIntegrator, x_pred, t)

    @unpack p, f, dt = integ
    @unpack jac, H!, InvPrecond = integ.constants
    @unpack u_pred, ddu, H = integ.cache
    PI = InvPrecond(dt)

    if !isnothing(jac)
        if isinplace(integ)
            jac(ddu, u_pred, p, t)
        else
            ddu .= jac(u_pred, p, t)
        end
        integ.destats.njacs += 1
    end
    H!(H, ddu)
    H .= H * PI
end

function measure!(integ, x_pred, t)
    measure_h!(integ, x_pred, t)
    measure_H!(integ, x_pred, t)

    @unpack R = integ.constants
    @unpack measurement, h, H = integ.cache

    v, S = measurement.μ, measurement.Σ
    v .= 0 .- h
    R .= Diagonal(eps.(v) .^ 2)
    S .= Symmetric(H * x_pred.Σ * H' .+ R)

    return nothing
end

function update!(integ::ODEFilterIntegrator, prediction)

    @unpack dt = integ
    @unpack R, q, d, Precond, InvPrecond, E1 = integ.constants
    @unpack measurement, h, H, K, x_filt = integ.cache
    P, PI = Precond(dt), InvPrecond(dt)

    v, S = measurement.μ, measurement.Σ

    m_p, P_p = prediction.μ, prediction.Σ

    S_inv = inv(S)
    K .= P_p * H' * S_inv

    x_filt.μ .= m_p .+ K*v
    # x_filt.μ .= m_p .+ P_p * H' * (S\v)

    # Vanilla
    # KSK = K*S*K'
    # KSK = P_p * H' * (S \ (H * P_p'))
    # x_filt.Σ .= P_p .- KSK
    # approx_diff!(x_filt.Σ, P_p, KSK)

    # Joseph Form
    x_filt.Σ .= Symmetric(X_A_Xt(PDMat(P_p), (I-K*H)))
    if !iszero(R)
        x_filt.Σ .+= Symmetric(X_A_Xt(PDMat(R), K))
    end

    assert_nonnegative_diagonal(x_filt.Σ)
    # cholesky(Symmetric(x_filt.Σ))

    return x_filt
end

function approx_diff!(A, B, C)
    @assert size(A) == size(B) == size(C)
    @assert eltype(A) == eltype(B) == eltype(C)
    # If B_ij ≈ C_ij, then A_ij = 0
    # But, only do this if the value in A is actually negative
    @simd for i in 1:length(A)
        @inbounds if B[i] ≈ C[i]
            A[i] = 0
        else
            A[i] = B[i] - C[i]
        end
    end
end


function fix_negative_variances(g::Gaussian, abstol::Real, reltol::Real)
    for i in 1:length(g.μ)
        if (g.Σ[i,i] < 0) && (sqrt(-g.Σ[i,i]) / (abstol + reltol * abs(g.μ[i]))) .< eps(eltype(g.Σ))
            # @info "fix_neg" g.Σ[i,i] g.μ[i] abstol reltol (abstol+reltol*abs(g.μ[i])) eps(eltype(g.Σ))*(abstol+reltol*abs(g.μ[i]))
            g.Σ[i,i] = eps(eltype(g.Σ))*(abstol+reltol*abs(g.μ[i]))
            g.Σ[i,i] = 0
        end
    end
end
