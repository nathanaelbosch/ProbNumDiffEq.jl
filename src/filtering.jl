"""Simple, understandable implementations for Gaussian filtering and smoothing

The functions all operate on `Gaussian` types.
"""


"""Vanilla PREDICT, without fancy checks or pre-allocation; use to test against"""
function predict(x_curr, Ah, Qh, PI=I)
    mean = Ah * x_curr.μ
    cov = Ah * x_curr.Σ * Ah' .+ Qh

    # zero_if_approx_similar!(cov, PI*cov*PI', zero(cov))
    # assert_good_covariance(cov)

    return Gaussian(mean, cov)
end


"""Vanilla UPDATE, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against
"""
function update(x_pred::Gaussian, h::AbstractVector, H::AbstractMatrix, R::AbstractMatrix, PI=I)
    m_p, P_p = x_pred.μ, x_pred.Σ

    # If the predicted covariance is zero, the prediction will not be adjusted!
    if all((P_p) .< eps(eltype(P_p)))
        return Gaussian(m_p, P_p)
    else
        v = 0 .- h
        S = Symmetric(H * P_p * H' .+ R)
        # K = P_p * H' * S_inv

        filt_mean = m_p .+ P_p * H' * (S\v)
        KSK = P_p * H' * (S \ (H * P_p'))
        filt_cov = P_p .- KSK

        zero_if_approx_similar!(filt_cov, P_p, KSK)
        # @info "update" filt_cov P_p KSK
        # @info "update" filt_cov P_p h H R S S_inv K KSK
        assert_good_covariance(filt_cov)

        return Gaussian(filt_mean, filt_cov)
    end
end


"""Vanilla SMOOTH, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against.
Requires the PREDICTed state.
"""
function smooth(x_curr::Gaussian, x_next_pred::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix,
                PI=I)
    # @info "smooth" x_curr x_next_pred x_next_smoothed Ah

    P_p_inv = inv(Symmetric(x_next_pred.Σ))
    Gain = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + Gain * (x_next_smoothed.μ - x_next_pred.μ)

    x_next_diff_cov = x_next_smoothed.Σ - x_next_pred.Σ
    # zero_if_approx_similar!(x_next_diff_cov, x_next_smoothed.Σ, x_next_pred.Σ)
    # zero_if_approx_similar!(x_next_diff_cov, PI*x_next_smoothed.Σ*PI', PI*x_next_pred.Σ*PI')
    GDG = Gain * (x_next_diff_cov) * Gain'
    smoothed_cov = x_curr.Σ + GDG

    # zero_if_approx_similar!(smoothed_cov, x_curr.Σ, -GDG)
    # zero_if_approx_similar!(smoothed_cov, PI*x_curr.Σ*PI', -PI*GDG*PI')

    try
        assert_good_covariance(smoothed_cov)
    catch e
        @error "Bad smoothed_cov" x_curr.Σ x_next_pred.Σ x_next_smoothed.Σ smoothed_cov GDG
        throw(e)
    end

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, Gain
end


function predsmooth(x_curr::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix, PI=I)
    x_next_pred = predict(x_curr, Ah, Qh, PI)
    x_curr_smoothed, _ = smooth(x_curr, x_next_pred, x_next_smoothed, Ah, PI)
    return x_curr_smoothed
end


function assert_good_covariance(cov)
    if !all(diag(cov) .>= 0)
        @error "Non-positive variances" cov diag(cov)
        error("The provided covariance has non-positive entries on the diagonal!")
    end
end
