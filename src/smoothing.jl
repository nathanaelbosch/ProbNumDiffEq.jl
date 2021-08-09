########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################
function smooth_all!(integ)
    integ.sol.x_smooth = copy(integ.sol.x_filt)

    @unpack A, Q = integ.cache
    @unpack x_smooth, t, diffusions = integ.sol
    x = x_smooth

    for i in length(x)-1:-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x[i], x[i+1])
            continue
        end

        make_preconditioners!(integ, dt)
        P, PI = integ.cache.P, integ.cache.PI

        Qh = apply_diffusion(Q, diffusions[i])

        x[i] = P * x[i]
        smooth!(x[i], P*x[i+1], A, Qh, integ)
        @assert !(any(isnan.(x[i].μ)) || any(isnan.(x[i].Σ))) "NaNs after smoothing"
        x[i] = PI * x[i]
    end
end


function smooth!(x_curr, x_next, Ah, Qh, integ)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    @unpack d, q = integ.cache
    @unpack x_tmp = integ.cache
    @unpack C1, G1, G2, C2 = integ.cache

    # Prediction: t -> t+1
    predict_mean!(x_tmp, x_curr, Ah, Qh)
    predict_cov!(x_tmp, x_curr, Ah, Qh, C1)

    # Smoothing
    P_p = x_tmp.Σ
    P_p_inv = inv(P_p)
    # G = x_curr.Σ * Ah' * P_p_inv
    G = mul!(G2, mul!(G1, x_curr.Σ, Ah'), P_p_inv)
    x_curr.μ .+= G * (x_next.μ .- x_tmp.μ)

    # Joseph-Form:
    M, L = C2.mat, C2.squareroot
    D = length(x_tmp.μ)
    mul!(view(L, 1:D, 1:D), (I-G*Ah), x_curr.Σ.squareroot)
    mul!(view(L, 1:D, D+1:2D), G, Qh.squareroot)
    mul!(view(L, 1:D, 2D+1:3D), G, x_next.Σ.squareroot)

    mul!(M, L, L')
    chol = cholesky!(Symmetric(M), check=false)

    if issuccess(chol)
        copy!(x_curr.Σ.squareroot, chol.U')
        mul!(x_curr.Σ.mat, chol.U', chol.U)
    else
        _, R = qr(L')
        copy!(x_curr.Σ.squareroot, R')
        mul!(x_curr.Σ.mat, R', R)
    end

    # _, P_s_R = qr(_R)
    # copy!(x_curr.Σ, SRMatrix(P_s_R'))

    assert_nonnegative_diagonal(x_curr.Σ)
    # PDMat(Symmetric(x_curr.Σ))

    return nothing
end
