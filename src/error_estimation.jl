abstract type AbstractErrorEstimator end

struct SchoberErrors <: AbstractErrorEstimator end
function estimate_errors(::SchoberErrors, integ)
    @unpack dt = integ
    @unpack InvPrecond = integ.constants
    @unpack σ_sq, Qh, H = integ.cache

    if σ_sq isa Real && isinf(σ_sq)
        return Inf
    end

    error_estimate = sqrt.(diag(H * σ_sq*Qh * H'))

    return error_estimate
end
