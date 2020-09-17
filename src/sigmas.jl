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
    @unpack d = integ.constants
    @unpack measurement = integ.cache

    v, S = measurement.μ, measurement.Σ
    sigma_t = v' * inv(S) * v / d

    if integ.success_iter == 0
        @assert length(integ.sigmas) == 0
        return sigma_t
    else
        @assert length(integ.sigmas) == integ.success_iter
        sigma_prev = integ.sigmas[end]
        sigma = sigma_prev + (sigma_t - sigma_prev) / integ.success_iter
        return sigma
    end
end


struct WeightedMLESigma <: AbstractStaticSigmaRule end
function static_sigma_estimation(rule::WeightedMLESigma, integ)
    @unpack d = integ.constants
    @unpack measurement = integ.cache

    v, S = measurement.μ, measurement.Σ
    sigma_t = v' * inv(S) * v / d

    if integ.success_iter == 0
        @assert length(integ.sigmas) == 0
        return sigma_t
    else
        @assert length(integ.sigmas) == integ.success_iter
        sigma_prev = integ.sigmas[end]
        sigma = (sigma_prev * integ.t + sigma_t * integ.dt) / integ.t_new
        return sigma
    end
end


"""Maximum a-posteriori sigma estimate when using an InverseGamma(1/2,1/2) prior

The mode of an InverseGamma(α,β) distribution is given by β/(α+1)
To compute this in an on-line basis from the previous sigma, we reverse the computation to
get the previous sum of residuals from sigma, and then modify that sum and compute the new
sigma.
"""
struct MAPSigma <: AbstractStaticSigmaRule end
function static_sigma_estimation(rule::MAPSigma, integ)
    @unpack d = integ.constants
    @unpack measurement = integ.cache

    N = integ.success_iter + 1
    v, S = measurement.μ, measurement.Σ
    res_t = v' * inv(S) * v / d

    α, β = 1/2, 1/2
    if integ.success_iter == 0
        @assert length(integ.sigmas) == 0
        sigma_t = (β + 1/2 * res_t) / (α + N*d/2 + 1)
        return sigma_t
    else
        @assert length(integ.sigmas) == integ.success_iter
        sigma_prev = integ.sigmas[end]
        res_prev = (sigma_prev * (α + (N-1)*d/2 + 1) - β) * 2
        res_sum_t = res_prev + res_t
        sigma = (β + 1/2 * res_sum_t) / (α + N*d/2 + 1)
        return sigma
    end
end


struct SchoberSigma <: AbstractDynamicSigmaRule end
function dynamic_sigma_estimation(kind::SchoberSigma, integ)
    @unpack d, R = integ.constants
    @unpack h, H, Qh = integ.cache
    @assert all(R .== 0) "The schober-sigma assumes R==0!"
    σ² = h' * inv(H*Qh*H') * h / d
    return σ²
end


"""Optimization-based sigma estimation

Basically, compute the arg-max of the evidence through an actual optimization routine!
Nice to know: The output "looks" very similar to the schober estimation!
But, my current benchmark shows that it's much much slower: ~300ms vs 13ms!

Update [15.09.2020]
I changed the code, from using the exp to computing sigma directly but thresholding to zero.
I also set sigma to exaclty zero for the cases where it's just too small, and I changed the
jitter to `eps`.

Note: It does not really seem to work that well!
"""
struct OptimSigma <: AbstractDynamicSigmaRule end
function dynamic_sigma_estimation(kind::OptimSigma, integ)
    @unpack d, q, R = integ.constants
    @unpack sigmas = integ
    @unpack h, H, Qh, x, Ah, σ_sq = integ.cache

    sigma_prev = sigma = length(sigmas) > 0 ? sigmas[end] : σ_sq

    z = h
    P = x.Σ

    # function negloglikelihood(σ_log)
    #     S = H*(Ah*P*Ah' + exp.(σ_log).* Qh)*H' + R
    #     S_inv = inv(S + 1e-12I)
    #     return logdet(S) + z' * S_inv * z
    # end
    # g!(x, storage) = (storage[1] = ForwardDiff.gradient(s -> negloglikelihood(s), x)[1])
    # res = optimize(negloglikelihood, [σ_sq], Newton())
    # σ² = exp(res.minimizer[1])

    # function neglikelihood(σ_log)
    #     S = H*(Ah*P*Ah' + exp.(σ_log).* Qh)*H' + R
    #     d = Gaussian(z, S)
    #     return - pdf(Gaussian(z, S), zeros(size(z)))
    # end
    # res = optimize(neglikelihood, [sigma_prev], Newton())
    # σ² = exp(res.minimizer[1])

    function negloglikelihood(σ)
        σ = max(σ, zero(σ))
        S = Symmetric(H*(Ah*P*Ah' + σ .* Qh)*H' + R)
        S_inv = inv(S + eps(eltype(S))*I)
        return logdet(S) + z' * S_inv * z
    end
    res = optimize(negloglikelihood, [one(σ_sq)], Newton())
    σ² = res.minimizer[1]

    (σ² < eps(typeof(σ²))) && (σ² = zero(σ²))  # set quasi-zero to exactly zero

    # @info "Optimization sigma:" σ²

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
    @unpack d, q, R = integ.constants
    @unpack sigmas = integ
    @unpack h, H, Qh, x, Ah, σ_sq = integ.cache

    sigma_prev = sigma = length(sigmas) > 0 ? sigmas[end] : σ_sq

    x_curr = x

    for i in 1:1
        # @info "EM-sigma" sigma_prev sigma i integ.t
        # @info "EM-sigma" h H integ.dt Qh Ah

        x_next_pred = predict(x_curr, Ah, sigma*Qh)
        x_next_filt = update(x_next_pred, h, H, R)
        x_curr_smoothed, Gain = smooth(x_curr, x_next_pred, x_next_filt, Ah)

        joint_distribution = Gaussian(
            [x_next_filt.μ
             x_curr_smoothed.μ],
            [x_next_filt.Σ         x_next_filt.Σ * Gain';
             Gain * x_next_filt.Σ  x_curr_smoothed.Σ]
        )

        A_tilde = [diagm(0=>ones(d*(q+1)))  -Ah]
        diff_mean = A_tilde * joint_distribution.μ
        diff_cov = A_tilde * joint_distribution.Σ * A_tilde'

        sigma = tr(Qh \ (diff_cov + diff_mean * diff_mean')) / (d*(q+1))
    end
    (sigma < 0) && (sigma=zero(sigma))
    # @info "em-sigma" sigma Qh integ.dt

    return sigma
end
