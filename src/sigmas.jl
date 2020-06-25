abstract type AbstractSigmaRule end
function static_sigma_estimation(rule::AbstractSigmaRule, solver, proposals)
    return 1
end
function dynamic_sigma_estimation(rule::AbstractSigmaRule; args...)
    return 1
end


struct MLESigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::MLESigma, solver, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = solver.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    σ² = mean(residuals)
    return σ²
end


struct WeightedMLESigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::WeightedMLESigma, solver, proposals)
    measurements = [p.measurement for p in accepted_proposals]
    d = solver.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    stepsizes = [p.dt for p in accepted_proposals]
    σ² = mean(residuals .* stepsizes)
    return σ²
end


struct MAPSigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::MAPSigma, solver, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = solver.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    N = length(residuals)

    α, β = 1/2, 1/2
    # prior = InverseGamma(α, β)
    α2, β2 = α + N*d/2, β + 1/2 * (sum(residuals))
    posterior = InverseGamma(α2, β2)
    sigma = mode(posterior)
    return sigma
end

struct Schober16Sigma <: AbstractSigmaRule end
function dynamic_sigma_estimation(rule::Schober16Sigma; H, Q, v, argv...)
    return v' * inv(H*Q*H') * v / length(v)
end
