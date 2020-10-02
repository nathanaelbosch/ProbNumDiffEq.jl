"""Simple, understandable implementations for Gaussian filtering and smoothing

The functions all operate on `Gaussian` types.
"""


"""Vanilla PREDICT, without fancy checks or pre-allocation; use to test against"""
function predict(x_curr, Ah, Qh, PI=I)
    mean = Ah * x_curr.μ
    cov = Ah * x_curr.Σ * Ah' .+ Qh

    return Gaussian(mean, cov)
end


"""Vanilla UPDATE, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against
"""
function update(x_pred::Gaussian, h::AbstractVector, H::AbstractMatrix, R::AbstractMatrix, PI=I)
    m_p, P_p = x_pred.μ, x_pred.Σ

    # If the predicted covariance is zero, the prediction will not be adjusted!
    v = 0 .- h
    S = Symmetric(H * P_p * H' .+ R)
    S_inv = inv(S)
    K = P_p * H' * S_inv

    filt_mean = m_p .+ P_p * H' * (S\v)
    # KSK = P_p * H' * (S \ (H * P_p'))
    # filt_cov = P_p .- KSK
    filt_cov = X_A_Xt(PDMat(Symmetric(P_p)), (I-K*H))
    if !iszero(R)
        filt_cov .+= Symmetric(X_A_Xt(PDMat(R), K))
    end

    assert_nonnegative_diagonal(filt_cov)

    return Gaussian(filt_mean, filt_cov)
end


"""Vanilla SMOOTH, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against.
Requires the PREDICTed state.
"""
function smooth(x_curr::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix,
                PI=I)
    # @info "smooth" x_curr x_next_pred x_next_smoothed Ah
    P_p = Symmetric(Ah * x_curr.Σ * Ah' .+ Qh)
    P_p_inv = inv(P_p)

    G = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - Ah*x_curr.μ)

    # x_next_diff_cov = x_next_smoothed.Σ - x_next_pred.Σ
    # GDG = Gain * (x_next_diff_cov) * Gain'
    # smoothed_cov = x_curr.Σ + GDG

    P = copy(x_curr.Σ)
    C_tilde = Ah
    K_tilde = P * Ah' * P_p_inv
    smoothed_cov = ((I - K_tilde*C_tilde) * P * (I - K_tilde*C_tilde)'
                    + K_tilde * Qh * K_tilde' + G * x_next_smoothed.Σ * G')

    assert_nonnegative_diagonal(smoothed_cov)

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end


function assert_nonnegative_diagonal(cov)
    if !all(diag(cov) .>= 0)
        @error "Non-positive variances" cov diag(cov)
        error("The provided covariance has non-positive entries on the diagonal!")
    end
end
