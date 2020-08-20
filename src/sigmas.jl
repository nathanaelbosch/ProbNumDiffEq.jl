abstract type AbstractSigmaRule end
function static_sigma_estimation(rule::AbstractSigmaRule, integ, proposals)
    return one(integ.cache.σ_sq)
end
function dynamic_sigma_estimation(rule::AbstractSigmaRule, integ)
    return one(integ.cache.σ_sq)
end


struct MLESigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::MLESigma, integ)
    @unpack proposals = integ.cache
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.constants.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    σ² = mean(residuals)
    return σ²
end


struct WeightedMLESigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::WeightedMLESigma, integ)
    @unpack proposals = integ.cache
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.constants.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    stepsizes = [p.dt for p in accepted_proposals]
    σ² = mean(residuals .* stepsizes)
    return σ²
end


struct MAPSigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::MAPSigma, integ)
    @unpack proposals = integ.cache
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.constants.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    N = length(residuals)

    α, β = 1/2, 1/2
    # prior = InverseGamma(α, β)
    α2, β2 = α + N*d/2, β + 1/2 * (sum(residuals))
    posterior = InverseGamma(α2, β2)
    sigma = mode(posterior)
    return sigma
end


struct SchoberSigma <: AbstractSigmaRule end
function dynamic_sigma_estimation(kind::SchoberSigma, integ)
    @unpack d = integ.constants
    @unpack h, H, Qh = integ.cache
    return h' * inv(H*Qh*H') * h / d
end
