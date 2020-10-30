"""Simple, understandable implementations for Gaussian filtering and smoothing

The functions all operate on `Gaussian` types.
"""


"""PREDICT

`predict!` is the in-place version, aiming to reduce allocations;
"""
function predict!(x_out, x_curr, Ah, Qh)
    mul!(x_out.μ, Ah, x_curr.μ)
    # TODO: This can be done more efficiently
    out_cov = X_A_Xt(x_curr.Σ, Ah)
    if !iszero(Qh)
        out_cov = out_cov + PSDMatrix(cholesky(Qh).L)
    end
    copy!(x_out.Σ, out_cov)
    return nothing
end
function predict(x_curr, Ah, Qh)
    x_out = copy(x_curr)
    predict!(x_out, x_curr, Ah, Qh)
    return x_out
end


"""Vanilla UPDATE, without fancy checks or pre-allocation

Use this to test any "advanced" implementation against
"""
function update(x_pred::Gaussian, h::AbstractVector, H::AbstractMatrix, R::AbstractMatrix)
    m_p, P_p = x_pred.μ, x_pred.Σ

    v = 0 .- h
    S = Symmetric(H * P_p * H' .+ R)
    S_inv = inv(S)
    K = P_p * H' * S_inv

    filt_mean = m_p .+ P_p * H' * (S\v)
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
function smooth(x_curr::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)
    x_pred = predict(x_curr, Ah, Qh)

    P_p = x_pred.Σ
    P_p_inv = inv(P_p)

    G = x_curr.Σ * Ah' * P_p_inv

    smoothed_mean = x_curr.μ + G * (x_next_smoothed.μ - x_pred.μ)

    K_tilde = x_curr.Σ * Ah' * P_p_inv
    smoothed_cov = (
        X_A_Xt(x_curr.Σ, (I - K_tilde*Ah))
        + X_A_Xt(PSDMatrix(cholesky(Qh).L), K_tilde)
        + X_A_Xt(x_next_smoothed.Σ, G)
    )

    assert_nonnegative_diagonal(smoothed_cov)

    x_curr_smoothed = Gaussian(smoothed_mean, smoothed_cov)
    return x_curr_smoothed, G
end
