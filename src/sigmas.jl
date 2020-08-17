abstract type AbstractSigmaRule end
function static_sigma_estimation(rule::AbstractSigmaRule, integ, proposals)
    return 1
end
function dynamic_sigma_estimation(rule::AbstractSigmaRule, integ)
    return 1
end


struct MLESigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::MLESigma, integ, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    σ² = mean(residuals)
    return σ²
end


struct WeightedMLESigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::WeightedMLESigma, integ, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    stepsizes = [p.dt for p in accepted_proposals]
    σ² = mean(residuals .* stepsizes)
    return σ²
end


struct MAPSigma <: AbstractSigmaRule end
function static_sigma_estimation(rule::MAPSigma, integ, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.d
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
function dynamic_sigma_estimation(rule::Schober16Sigma, integ)
    @unpack d = integ.constants
    @unpack h, H, Qh = integ.cache
    return h' * inv(H*Qh*H') * h / d
end


# using Optim
# struct Schober16SigmaGlobal <: AbstractSigmaRule end
# function dynamic_sigma_estimation(rule::Schober16SigmaGlobal; H, Q, v, P, A, R, argv...)

#     """p(z|σ²)"""
#     function sigma_to_pz(σ²)
#         s = sum(σ²)
#         P_p = A*P*A' + s*Q
#         S = H * P_p * H' + R
#         return v' * inv(S) * v / length(v)
#     end

#     results = Optim.optimize(sigma_to_pz, [1.], Newton(); autodiff=:forward)

#     @show sigma_to_pz(1)
#     @show results.minimizer

#     out = v' * inv(H*Q*H') * v / length(v)
#     @show out
#     return out
# end
