abstract type AbstractErrorEstimator end

struct SchoberErrors <: AbstractErrorEstimator end
function estimate_errors(::SchoberErrors, integ)
    @unpack dt = integ
    @unpack InvPrecond = integ.constants
    @unpack σ_sq, Qh, H = integ.cache

    PI = InvPrecond(dt)
    H = H * PI

    error_estimate = sqrt.(diag(H * σ_sq*Qh * H'))

    return error_estimate
end


struct PredictionErrors <: AbstractErrorEstimator end
function estimate_errors(::PredictionErrors, integ)
    @unpack dt = integ
    @unpack E0, InvPrecond = integ.constants
    @unpack σ_sq, Qh = integ.cache
    PI = InvPrecond(dt)

    f_cov = σ_sq * E0 * PI * Qh * PI' * E0'
    error_estimate = sqrt.(diag(f_cov))

    return error_estimate
end



struct FilterErrors <: AbstractErrorEstimator end
function estimate_errors(::FilterErrors, integ)
    @unpack dt = integ
    @unpack E0, R, InvPrecond = integ.constants
    @unpack σ_sq, Qh, H = integ.cache
    PI = InvPrecond(dt)

    if iszero(Qh) || iszero(σ_sq)
        return zero(integ.EEst)
    end

    P_pred_loc = σ_sq .* Qh

    H = H * PI
    S_loc = Symmetric(H * P_pred_loc * H' .+ R)
    K_loc = P_pred_loc * H' * inv(S_loc)
    P_filt_loc = P_pred_loc .- K_loc * S_loc * K_loc'

    f_filt_cov = E0 * PI * P_filt_loc * PI * E0'
    error_estimate = sqrt.(diag(f_filt_cov))

    return error_estimate
end
