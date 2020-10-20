abstract type AbstractSigmaRule end
abstract type AbstractStaticSigmaRule <: AbstractSigmaRule end
abstract type AbstractDynamicSigmaRule <: AbstractSigmaRule end
isstatic(sigmarule::AbstractStaticSigmaRule) = true
isdynamic(sigmarule::AbstractStaticSigmaRule) = false
isstatic(sigmarule::AbstractDynamicSigmaRule) = false
isdynamic(sigmarule::AbstractDynamicSigmaRule) = true
initial_sigma(sigmarule::AbstractSigmaRule, d, q) = 1.0


struct MLESigma <: AbstractStaticSigmaRule end
function sigma_estimation(rule::MLESigma, integ)
    @unpack d = integ.cache
    @unpack measurement = integ.cache

    v, S = measurement.μ, measurement.Σ

    if iszero(v)
        return zero(integ.cache.σ_sq)
    end
    if iszero(S)
        return Inf
    end

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


"""Maximum a-posteriori sigma estimate when using an InverseGamma(1/2,1/2) prior

The mode of an InverseGamma(α,β) distribution is given by β/(α+1)
To compute this in an on-line basis from the previous sigma, we reverse the computation to
get the previous sum of residuals from sigma, and then modify that sum and compute the new
sigma.
"""
struct MAPSigma <: AbstractStaticSigmaRule end
function sigma_estimation(rule::MAPSigma, integ)
    @unpack d = integ.cache
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
function sigma_estimation(kind::SchoberSigma, integ)
    @unpack d, R = integ.cache
    @unpack H, Qh, measurement = integ.cache
    # @assert all(R .== 0) "The schober-sigma assumes R==0!"
    z = measurement.μ
    σ² = z' * inv(H*Qh*H') * z / d
    return σ²
end


struct MVSchoberSigma <: AbstractDynamicSigmaRule end
initial_sigma(sigmarule::MVSchoberSigma, d, q) = kron(ones(q+1, q+1), diagm(0 => ones(d)))
function sigma_estimation(kind::MVSchoberSigma, integ)
    @unpack dt = integ
    @unpack d, q, R, InvPrecond, E1 = integ.cache
    @unpack H, Qh, measurement = integ.cache
    z = measurement.μ

    # @assert all(R .== 0) "The schober-sigma assumes R==0!"

    # Assert EKF0
    PI = InvPrecond(dt)
    @assert all(H .== E1 * PI)

    # More safety checks
    @assert isdiag(H*Qh*H')
    @assert length(unique(diag(H*Qh*H'))) == 1
    Q0_11 = diag(H*Qh*H')[1]

    Σ_ii = z .^ 2 ./ Q0_11
    Σ_ii .= max.(Σ_ii, eps(eltype(Σ_ii)))
    Σ = Diagonal(Σ_ii)

    Σ_out = kron(ones(q+1, q+1), Σ)
    # @info "MVSchober sigma" Σ Σ_out
    return Σ_out
end


struct MVMLESigma <: AbstractStaticSigmaRule end
initial_sigma(sigmarule::MVMLESigma, d, q) = kron(ones(q+1, q+1), diagm(0 => ones(d)))
function sigma_estimation(kind::MVMLESigma, integ)
    @unpack dt = integ
    @unpack d, q, R, InvPrecond, E1 = integ.cache
    @unpack measurement, H = integ.cache

    # Assert EKF0
    PI = InvPrecond(dt)
    @assert all(H .== E1 * PI)

    @unpack measurement = integ.cache
    v, S = measurement.μ, measurement.Σ

    # More safety checks
    @assert isdiag(S)
    # @assert length(unique(diag(S))) == 1
    S_11 = diag(S)[1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(ones(q+1, q+1), Σ)
    # @info "MV-MLE-Sigma" v S Σ Σ_out

    if integ.success_iter == 0
        @assert length(integ.sigmas) == 0
        return Σ_out
    else
        @assert length(integ.sigmas) == integ.success_iter
        sigma_prev = integ.sigmas[end]
        sigma = sigma_prev + (Σ_out - sigma_prev) / integ.success_iter
        return sigma
    end
end
