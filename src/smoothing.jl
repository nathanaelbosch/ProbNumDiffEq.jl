########################################################################################
# Post-Processing: Smoothing and uncertainty calibration
########################################################################################
function smooth_all!(integ)
    integ.sol.x_smooth = copy(integ.sol.x_filt)

    @unpack A, Q = integ.cache
    @unpack x_smooth, t, diffusions = integ.sol
    @unpack x_tmp, x_tmp2 = integ.cache
    x = x_smooth

    for i in length(x)-1:-1:1
        dt = t[i+1] - t[i]
        if iszero(dt)
            copy!(x[i], x[i+1])
            continue
        end

        make_preconditioners!(integ.cache, dt)
        P, PI = integ.cache.P, integ.cache.PI

        mul!(x_tmp, P, x[i])
        mul!(x_tmp2, P, x[i+1])
        smooth!(x_tmp, x_tmp2, A, Q, integ, diffusions[i])
        mul!(x[i], PI, x_tmp)
    end
end


function smooth!(x_curr, x_next, Ah, Qh, integ, diffusion=1)
    # x_curr is the state at time t_n (filter estimate) that we want to smooth
    # x_next is the state at time t_{n+1}, already smoothed, which we use for smoothing
    @unpack d, q = integ.cache
    @unpack x_pred = integ.cache
    @unpack C1, G1, G2, C2 = integ.cache

    # Prediction: t -> t+1
    predict_mean!(x_pred, x_curr, Ah)
    predict_cov!(x_pred, x_curr, Ah, Qh, C1, diffusion)

    # Smoothing
    # TODO Change the following to `_matmul!`
    # G = x_curr.Σ * Ah' * P_p_inv
    P_p_chol = Cholesky(x_pred.Σ.squareroot, :L, 0)
    G = rdiv!(_matmul!(G1, x_curr.Σ.mat, Ah'), P_p_chol)

    x_curr.μ .+= G * (x_next.μ .- x_pred.μ)

    # Joseph-Form:
    M, L = C2.mat, C2.squareroot
    D = length(x_pred.μ)

    _matmul!(G2, G, Ah)
    copy!(view(L, 1:D, 1:D), x_curr.Σ.squareroot)
    _matmul!(view(L, 1:D, 1:D), G2, x_curr.Σ.squareroot, -1.0, 1.0)

    _matmul!(view(L, 1:D, D+1:2D), _matmul!(G2, G, sqrt.(diffusion)), Qh.squareroot)
    _matmul!(view(L, 1:D, 2D+1:3D), G, x_next.Σ.squareroot)

    # _matmul!(M, L, L')
    # chol = cholesky!(Symmetric(M), check=false)

    if false && issuccess(chol)
        copy!(x_curr.Σ.squareroot, chol.U')
        mul!(x_curr.Σ.mat, chol.U', chol.U)
    elseif eltype(L) <: Union{Float16, Float32, Float64}
        Q = lq!(L)
        QL = Q.L
        copy!(x_curr.Σ.squareroot, QL)
        _matmul!(x_curr.Σ.mat, QL, QL')
    else
        Q = qr(L')
        QL = Q.R'
        copy!(x_curr.Σ.squareroot, QL)
        _matmul!(x_curr.Σ.mat, QL, QL')
    end

    return nothing
end
