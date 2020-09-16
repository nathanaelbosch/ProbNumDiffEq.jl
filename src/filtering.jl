"""Simple, understandable implementations for Gaussian filtering and smoothing

The functions all operate on `Gaussian` types.
"""


"""Vanilla PREDICT, without fancy checks or pre-allocation; use to test against"""
function predict(x_curr, Ah, Qh)
    mean = Ah * x_curr.μ
    cov = Ah * x_curr.Σ * Ah' .+ Qh
    assert_good_covariance(cov)
    return Gaussian(mean, cov)
end


"""Vanilla UPDATE, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against
"""
function update(x_pred::Gaussian, h::AbstractVector, H::AbstractMatrix, R::AbstractMatrix)
    m_p, P_p = x_pred.μ, x_pred.Σ

    # If the predicted covariance is zero, the prediction will not be adjusted!
    if all(P_p .== 0)
        return Gaussian(m_p, P_P)
    else
        v = 0 .- h
        S = Symmetric(H * P_p * H' .+ R)
        S_inv = inv(S)
        K = P_p * H' * S_inv

        filt_mean = m_p .+ K*v
        KSK = K*S*K'
        filt_cov = P_p .- KSK

        zero_if_approx_similar!(filt_cov, P_p, KSK)
        assert_good_covariance(filt_cov)

        return Gaussian(filt_mean, filt_cov)
    end
end


"""Vanilla SMOOTH, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against.
Requires the PREDICTed state.
"""
function smooth(x_curr::Gaussian, x_next_pred::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix)
    # @info "smooth" x_curr x_next_pred x_next_smoothed Ah

    P_p_inv = inv(Symmetric(x_next_pred.Σ))
    Gain = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + Gain * (x_next_smoothed.μ - x_next_pred.μ)
    GDG = Gain * (x_next_smoothed.Σ - x_next_pred.Σ) * Gain'
    smoothed_cov = x_curr.Σ + GDG

    zero_if_approx_similar!(smoothed_cov, x_curr.Σ, -GDG)

    try
        assert_good_covariance(smoothed_cov)
    catch e
        @error "Bad smoothed_cov" x_curr.Σ x_next_pred.Σ x_next_smoothed.Σ smoothed_cov GDG
        throw(e)
    end

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, Gain
end


function assert_good_covariance(cov)
    if !all(diag(cov) .>= 0)
        @error "Non-positive variances" cov
        error("The provided covariance has non-positive entries on the diagonal!")
    end
end
