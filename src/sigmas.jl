abstract type AbstractSigmaRule end
abstract type AbstractStaticSigmaRule <: AbstractSigmaRule end
abstract type AbstractDynamicSigmaRule <: AbstractSigmaRule end
# function static_sigma_estimation(rule::AbstractDynamicSigmaRule, integ)
#     return one(integ.cache.σ_sq)
# end
# function dynamic_sigma_estimation(rule::AbstractStaticSigmaRule, integ)
#     return one(integ.cache.σ_sq)
# end
isstatic(sigmarule::AbstractStaticSigmaRule) = true
isdynamic(sigmarule::AbstractStaticSigmaRule) = false
isstatic(sigmarule::AbstractDynamicSigmaRule) = false
isdynamic(sigmarule::AbstractDynamicSigmaRule) = true


struct MLESigma <: AbstractStaticSigmaRule end
function static_sigma_estimation(rule::MLESigma, integ)
    @unpack proposals = integ
    @unpack d = integ.constants
    @unpack measurement = integ.cache

    v, S = measurement.μ, measurement.Σ
    sigma_t = v' * inv(S) * v / d

    if integ.iter == 1
        @assert length(integ.sigmas) == 0
        return sigma_t
    else
        @assert length(integ.sigmas)+1 == integ.iter
        sigma_prev = integ.sigmas[end]
        sigma = sigma_prev + (sigma_t - sigma_prev) / integ.iter
        return sigma
    end
end


struct WeightedMLESigma <: AbstractStaticSigmaRule end
function static_sigma_estimation(rule::WeightedMLESigma, integ)
    @warn "WeightedMLESigma is implemented VERY inefficiently right now"
    @unpack proposals = integ
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = integ.constants.d
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    stepsizes = [p.dt for p in accepted_proposals]
    σ² = mean(residuals .* stepsizes)
    return σ²
end


struct MAPSigma <: AbstractStaticSigmaRule end
function static_sigma_estimation(rule::MAPSigma, integ)
    @warn "MAPSigma is implemented VERY inefficiently right now"
    @unpack proposals = integ
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


struct SchoberSigma <: AbstractDynamicSigmaRule end
function dynamic_sigma_estimation(kind::SchoberSigma, integ)
    @unpack d = integ.constants
    @unpack h, H, Qh = integ.cache
    jitter = 1e-12
    σ² = h' * inv(H*Qh*H' + jitter*I) * h / d
    return σ²
end


"""Optimization-based sigma estimation

Basically, compute the arg-max of the evidence through an actual optimization routine!
Nice to know: The output "looks" very similar to the schober estimation!
But, my current benchmark shows that it's much much slower: ~300ms vs 13ms!
"""
struct OptimSigma <: AbstractDynamicSigmaRule end
function dynamic_sigma_estimation(kind::OptimSigma, integ)
    @unpack d, q, R = integ.constants
    @unpack h, H, Qh, x, Ah, σ_sq = integ.cache

    z = h
    P = x.Σ

    function negloglikelihood(σ_log)
        S = H*(Ah*P*Ah' + exp.(σ_log).* Qh)*H' + R
        S_inv = inv(S + 1e-12I)
        return logdet(S) + z' * S_inv * z
    end
    # g!(x, storage) = (storage[1] = ForwardDiff.gradient(s -> negloglikelihood(s), x)[1])
    res = optimize(negloglikelihood, [σ_sq], Newton())

    σ² = exp(res.minimizer[1])
    return σ²
end


"""Filip's proposition: Estimate sigma through a one-step EM

This seems pretty stable! One iteration indeed feels like it is enough.
I compared this single loop approach with one with `i in 1:1000`, and could not notice a difference.
=> This seems cool!
It does not seem to behave too different from the schober sigmas, but I mean the theory is wayy nicer!
"""
struct EMSigma <: AbstractDynamicSigmaRule end
function dynamic_sigma_estimation(kind::EMSigma, integ)
    @unpack d, q = integ.constants
    @unpack h, H, Qh, x_pred, x, Ah, σ_sq = integ.cache
    @unpack R = integ.constants

    sigma = σ_sq

    x_prev = x

    for i in 1:1
        x_n_pred = Gaussian(Ah * x_prev.μ, Ah * x_prev.Σ * Ah' + sigma*Qh)

        _m, _P = x_n_pred.μ, x_n_pred.Σ
        S = H * _P * H' + R
        K = _P * H' * inv(S)
        x_n_filt = Gaussian(_m + K*h, _P - K*S*K')

        # x_prev = integ.state_estimates[end]

        _m, _P = kf_smooth(x_prev.μ, x_prev.Σ, x_n_pred.μ, x_n_pred.Σ, x_n_filt.μ, x_n_filt.Σ, Ah, sigma*Qh)
        x_prev_smoothed = Gaussian(_m, _P)

        # Compute σ² in closed form:
        diff = x_n_filt.μ - Ah*x_prev_smoothed.μ
        sigma = diff' * inv(Qh) * diff / (d*(q+1))
    end

    return sigma
end
