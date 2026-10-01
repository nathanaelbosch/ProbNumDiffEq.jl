"""
Check the correctness of the filtering implementations vs. basic readable math code
"""

using Test
using ProbNumDiffEq
using LinearAlgebra
import ProbNumDiffEq: IsometricKroneckerProduct, BlocksOfDiagonals
import ProbNumDiffEq as PNDE
using FillArrays
import ProbNumDiffEq: logpdf

@testset "PREDICT" begin
    # Setup
    d = 2
    q = 2
    D = d * (q + 1)
    m = rand(D)

    _P_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q + 1, q + 1))))
    _P = _P_R'_P_R
    PM = Matrix(_P)

    _A = IsometricKroneckerProduct(d, rand(q + 1, q + 1))
    AM = Matrix(_A)

    _Q_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q + 1, q + 1))))
    _Q = _Q_R'_Q_R
    QM = Matrix(_Q)

    # PREDICT
    m_p = AM * m
    P_p = AM * PM * AM' + QM

    @testset "Factorization: $_FAC" for _FAC in (
        PNDE.DenseCovariance,
        PNDE.BlockDiagonalCovariance,
        PNDE.IsometricKroneckerCovariance,
    )
        FAC = _FAC{Float64}(d, q)

        P_R = PNDE.to_factorized_matrix(FAC, _P_R)
        P = P_R'P_R
        A = PNDE.to_factorized_matrix(FAC, _A)
        Q_R = PNDE.to_factorized_matrix(FAC, _Q_R)
        Q = Q_R'Q_R

        x_curr = Gaussian(m, P)
        x_out = copy(x_curr)

        C_DxD = PNDE.factorized_zeros(FAC, D, D)
        C_2DxD = PNDE.factorized_zeros(FAC, 2D, D)
        C_3DxD = PNDE.factorized_zeros(FAC, 3D, D)

        @testset "predict" begin
            x_out = ProbNumDiffEq.predict(x_curr, A, Q)
            @test m_p ≈ x_out.μ
            @test P_p ≈ x_out.Σ
        end

        @testset "predict! with PSDMatrix" begin
            x_curr = Gaussian(m, PSDMatrix(P_R))
            x_out = copy(x_curr)
            Q_SR = PSDMatrix(Q_R)
            ProbNumDiffEq.predict!(x_out, x_curr, A, Q_SR, C_DxD, C_2DxD)
            @test m_p ≈ x_out.μ
            @test P_p ≈ Matrix(x_out.Σ)
        end

        @testset "predict! with PSDMatrix and diffusion" begin
            for diffusion in (rand(), rand() * I(d), Diagonal(rand(d)))
                if _FAC == PNDE.IsometricKroneckerCovariance && !(diffusion isa Number)
                    continue
                end
                _diffusions = diffusion isa Number ? diffusion * Ones(d) : diffusion.diag

                QM_diff = Matrix(BlocksOfDiagonals([σ² * _Q.B for σ² in _diffusions]))
                P_p_diff = AM * PM * AM' + QM_diff

                x_curr = Gaussian(m, PSDMatrix(P_R))
                x_out = copy(x_curr)
                Q_SR = PSDMatrix(Q_R)
                ProbNumDiffEq.predict!(x_out, x_curr, A, Q_SR, C_DxD, C_2DxD, diffusion)
                @test P_p_diff ≈ Matrix(x_out.Σ)
            end
        end

        @testset "predict! with zero diffusion" begin
            x_curr = Gaussian(m, PSDMatrix(P_R))
            x_out = copy(x_curr)
            Q_SR = PSDMatrix(Q_R)
            ProbNumDiffEq.predict!(x_out, x_curr, A, Q_SR, C_DxD, C_2DxD, 0)
            @test m_p ≈ x_out.μ
            @test Matrix(x_out.Σ) ≈ Matrix(X_A_Xt(x_curr.Σ, A))
        end

        @testset "predict with kernel and marginalize!" begin
            x_curr = Gaussian(m, PSDMatrix(P_R))
            x_out = copy(x_curr)
            # marginalize! needs tall square-roots:
            Q_SR = if Q_R isa IsometricKroneckerProduct
                PSDMatrix(IsometricKroneckerProduct(Q_R.rdim, [Q_R.B; zero(Q_R.B)]))
            elseif Q_R isa BlocksOfDiagonals
                PSDMatrix(BlocksOfDiagonals([[B; zero(B)] for B in Q_R.blocks]))
            else
                PSDMatrix([Q_R; zero(Q_R)])
            end
            K = ProbNumDiffEq.AffineNormalKernel(A, Q_SR)
            T = eltype(m)
            ProbNumDiffEq.marginalize!(x_out, x_curr, K; C_DxD, C_3DxD)
            @test m_p ≈ x_out.μ
            @test P_p ≈ Matrix(x_out.Σ)
        end
    end
end

@testset "UPDATE" begin
    # Setup
    d = 5
    o = 1

    m_p = rand(d)
    _P_p_R = IsometricKroneckerProduct(o, Matrix(UpperTriangular(rand(d, d))))
    _P_p = _P_p_R'_P_p_R
    P_p_M = Matrix(_P_p)

    # Measure
    _HB = rand(1, d)
    _H = IsometricKroneckerProduct(o, _HB)
    HM = Matrix(_H)
    R = zeros(o, o)

    z_data = zeros(o)
    z = _H * m_p
    _SR = _P_p_R * _H'
    _S = _SR'_SR
    SM = Matrix(_S)

    LL = logpdf(Gaussian(z, SM), z_data)

    # UPDATE
    KM = P_p_M * HM' / SM
    m = m_p + KM * (z_data .- z)
    P = P_p_M - KM * SM * KM'

    _R_R = rand(o, o)
    _R = _R_R'_R_R
    _SR_noisy = qr([_P_p_R * _H'; _R_R]).R |> Matrix
    _S_noisy = _SR_noisy'_SR_noisy
    SM_noisy = Matrix(_S_noisy)

    KM_noisy = P_p_M * HM' / SM_noisy
    m_noisy = m_p + KM_noisy * (z_data .- z)
    P_noisy = P_p_M - KM_noisy * SM_noisy * KM_noisy'

    @testset "Factorization: $_FAC" for _FAC in (
        PNDE.DenseCovariance,
        PNDE.BlockDiagonalCovariance,
        PNDE.IsometricKroneckerCovariance,
    )
        FAC = _FAC{Float64}(o, d)

        C_dxd = PNDE.factorized_zeros(FAC, o, o)
        C_d = zeros(o)
        C_Dxd = PNDE.factorized_zeros(FAC, d, o)
        C_DxD = PNDE.factorized_zeros(FAC, d, d)
        C_2DxD = PNDE.factorized_zeros(FAC, 2d, d)
        C_3DxD = PNDE.factorized_zeros(FAC, 3d, d)

        P_p_R = PNDE.to_factorized_matrix(FAC, _P_p_R)
        P_p = P_p_R'P_p_R

        H = PNDE.to_factorized_matrix(FAC, _H)

        SR = PNDE.to_factorized_matrix(FAC, _SR)
        S = SR'SR

        x_pred = Gaussian(m_p, P_p)
        x_out = copy(x_pred)
        measurement = Gaussian(z, S)

        @testset "update" begin
            x_out = ProbNumDiffEq.update(x_pred, measurement, H)
            @test m ≈ x_out.μ
            @test P ≈ x_out.Σ
        end

        @testset "update!" begin
            @testset "PSDMatrix" begin
                K_cache = copy(C_Dxd)
                K2_cache = copy(C_Dxd)
                M_cache = C_DxD
                S = measurement.Σ
                msmnt = Gaussian(measurement.μ, PSDMatrix(SR))
                O_cache = C_dxd
                z_cache = C_d
                x_pred = Gaussian(x_pred.μ, PSDMatrix(P_p_R))
                x_out = copy(x_pred)
                _, ll = ProbNumDiffEq.update!(
                    x_out,
                    x_pred,
                    msmnt,
                    H,
                    K_cache,
                    K2_cache,
                    M_cache,
                    O_cache,
                    z_cache,
                )
                @test m ≈ x_out.μ
                @test P ≈ Matrix(x_out.Σ)
                @test ll ≈ LL
            end
            @testset "Zero predicted covariance" begin
                K_cache = copy(C_Dxd)
                K2_cache = copy(C_Dxd)
                M_cache = C_DxD
                S = measurement.Σ
                msmnt = Gaussian(measurement.μ, PSDMatrix(SR))
                O_cache = C_dxd
                z_cache = C_d
                x_pred = Gaussian(x_pred.μ, PSDMatrix(zero(P_p_R)))
                x_out = copy(x_pred)
                ProbNumDiffEq.update!(
                    x_out,
                    x_pred,
                    msmnt,
                    H,
                    K_cache,
                    K2_cache,
                    M_cache,
                    O_cache,
                    z_cache,
                )
                @test x_out == x_pred
            end
        end
    end
end

@testset "UPDATE of a subset of blocks" begin
    # The state blocks that are not observed keep their predicted values, also when
    # `x_out` does not start as a copy of `x_pred`
    d, q, obs_dims = 3, 2, [3, 1]
    o = length(obs_dims)
    blockdiag(f, n) = BlocksOfDiagonals([f() for _ in 1:n])
    uppertri() = Matrix(UpperTriangular(rand(q + 1, q + 1)))
    x_pred = Gaussian(rand(d * (q + 1)), PSDMatrix(blockdiag(uppertri, d)))
    msmnt = Gaussian(rand(o), PSDMatrix(blockdiag(() -> rand(1, 1) .+ 1, o)))
    H = blockdiag(() -> rand(1, q + 1), o)
    FAC = PNDE.BlockDiagonalCovariance{Float64}(o, q)
    D_o = o * (q + 1)
    caches = (
        PNDE.factorized_zeros(FAC, D_o, o),
        PNDE.factorized_zeros(FAC, D_o, o),
        PNDE.factorized_zeros(FAC, D_o, D_o),
        PNDE.factorized_zeros(FAC, o, o),
        zeros(o),
    )

    x_out = copy(x_pred)
    _, ll = PNDE.update!(x_out, x_pred, msmnt, PNDE.BlockSelection(H, obs_dims), caches...)
    x_out_zero = Gaussian(zero(x_pred.μ), PSDMatrix(zero(x_pred.Σ.R)))
    _, ll_zero = PNDE.update!(
        x_out_zero, x_pred, msmnt, PNDE.BlockSelection(H, obs_dims), caches...)
    @test x_out_zero == x_out
    @test ll_zero == ll

    # Dense reference: measurement `k` observes state block `obs_dims[k]`
    H_dense = zeros(o, d * (q + 1))
    for (k, i) in enumerate(obs_dims)
        H_dense[k, i:d:end] .= vec(H.blocks[k])
    end
    P, S = Matrix(x_pred.Σ), Matrix(msmnt.Σ)
    K = P * H_dense' / S
    @test x_out.μ ≈ x_pred.μ - K * msmnt.μ
    @test Matrix(x_out.Σ) ≈ (I - K * H_dense) * P * (I - K * H_dense)'
    @test ll ≈ logpdf(Gaussian(msmnt.μ, S), zeros(o))
end

@testset "UPDATE with observation noise" begin
    # Setup
    d = 5
    m_p = rand(d)
    P_p_R = Matrix(UpperTriangular(rand(d, d)))
    P_p = P_p_R'P_p_R

    # Measure
    o = 1
    _HB = rand(1, d)
    H = kron(I(o), _HB)
    R_R = rand(o, o)
    R = R_R'R_R

    z_data = zeros(o)
    z = H * m_p
    SR = qr([P_p_R * H'; R_R]).R |> Matrix
    S = Symmetric(SR'SR)

    LL = logpdf(Gaussian(z, S), z_data)

    # UPDATE
    S_inv = inv(S)
    K = P_p * H' * S_inv
    m = m_p + K * (z_data .- z)
    P = P_p - K * S * K'

    x_pred = Gaussian(m_p, P_p)
    x_out = copy(x_pred)
    measurement = Gaussian(z, S)

    C_dxd = zeros(o, o)
    C_d = zeros(o)
    C_Dxd = zeros(d, o)
    C_DxD = zeros(d, d)
    C_2DxD = zeros(2d, d)
    C_3DxD = zeros(3d, d)

    _fstr(F) = F ? "Kronecker" : "None"
    @testset "Factorization: $(_fstr(KRONECKER))" for KRONECKER in (false, true)
        if KRONECKER
            P_p_R = IsometricKroneckerProduct(1, P_p_R)
            P_p = P_p_R'P_p_R

            H = IsometricKroneckerProduct(1, _HB)
            R_R = IsometricKroneckerProduct(1, R_R)
            R = R'R

            SR = IsometricKroneckerProduct(1, SR)
            S = SR'SR

            x_pred = Gaussian(m_p, P_p)
            x_out = copy(x_pred)
            measurement = Gaussian(z, S)

            C_dxd = IsometricKroneckerProduct(1, C_dxd)
            C_Dxd = IsometricKroneckerProduct(1, C_Dxd)
            C_DxD = IsometricKroneckerProduct(1, C_DxD)
            C_2DxD = IsometricKroneckerProduct(1, C_2DxD)
            C_3DxD = IsometricKroneckerProduct(1, C_3DxD)
        end

        @testset "update" begin
            x_out = ProbNumDiffEq.update(x_pred, measurement, H)
            @test m ≈ x_out.μ
            @test P ≈ x_out.Σ
        end

        @testset "update!" begin
            @testset "PSDMatrix" begin
                K_cache = copy(C_Dxd)
                K2_cache = copy(C_Dxd)
                M_cache = C_DxD
                S = measurement.Σ
                msmnt = Gaussian(measurement.μ, PSDMatrix(SR))
                O_cache = C_dxd
                z_cache = C_d
                x_pred = Gaussian(x_pred.μ, PSDMatrix(P_p_R))
                x_out = copy(x_pred)
                _, ll = ProbNumDiffEq.update!(
                    x_out,
                    x_pred,
                    msmnt,
                    H,
                    K_cache,
                    K2_cache,
                    M_cache,
                    O_cache,
                    z_cache;
                    R=PSDMatrix(R_R),
                )
                @test m ≈ x_out.μ
                @test P ≈ Matrix(x_out.Σ)
                @test ll ≈ LL
            end
            @testset "Zero predicted covariance" begin
                K_cache = copy(C_Dxd)
                K2_cache = copy(C_Dxd)
                M_cache = C_DxD
                S = measurement.Σ
                msmnt = Gaussian(measurement.μ, PSDMatrix(SR))
                O_cache = C_dxd
                z_cache = C_d
                x_pred = Gaussian(x_pred.μ, PSDMatrix(zero(P_p_R)))
                x_out = copy(x_pred)
                ProbNumDiffEq.update!(
                    x_out,
                    x_pred,
                    msmnt,
                    H,
                    K_cache,
                    K2_cache,
                    M_cache,
                    O_cache,
                    z_cache,
                    R=PSDMatrix(R_R),
                )
                @test x_out == x_pred
            end
            @testset "Zero predicted covariance and zero noise" begin
                x_pred = Gaussian(x_pred.μ, PSDMatrix(zero(P_p_R)))
                x_out = copy(x_pred)
                _, ll = ProbNumDiffEq.update!(
                    x_out,
                    x_pred,
                    Gaussian(measurement.μ, PSDMatrix(SR)),
                    H,
                    copy(C_Dxd),
                    copy(C_Dxd),
                    C_DxD,
                    C_dxd,
                    C_d,
                    R=PSDMatrix(zero(R_R)),
                )
                @test x_out == x_pred
                @test ll == -Inf
            end
        end
    end
end

@testset "UPDATE log-likelihood with a multi-dimensional Kronecker covariance" begin
    # `S ⊗ I_d` has the log-determinant `d * logdet(S)`
    d, q1 = 3, 4
    FAC = PNDE.IsometricKroneckerCovariance{Float64}(d, q1 - 1)
    D = d * q1
    P_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q1, q1))))
    H = IsometricKroneckerProduct(d, rand(1, q1))
    SR = IsometricKroneckerProduct(d, Matrix(qr(P_R.B * H.B').R))
    m_p = rand(D)
    z = H * m_p

    x_pred = Gaussian(m_p, PSDMatrix(P_R))
    x_out = copy(x_pred)
    _, ll = ProbNumDiffEq.update!(
        x_out,
        x_pred,
        Gaussian(z, PSDMatrix(SR)),
        H,
        PNDE.factorized_zeros(FAC, D, d),
        PNDE.factorized_zeros(FAC, D, d),
        PNDE.factorized_zeros(FAC, D, D),
        PNDE.factorized_zeros(FAC, d, d),
        zeros(d),
    )
    @test ll ≈ logpdf(Gaussian(z, Matrix(SR'SR)), zeros(d))
end

@testset "SMOOTH" begin
    # Setup
    d = 5
    q = 2
    D = d * (q + 1)

    m, m_s = rand(D), rand(D)
    _P_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q + 1, q + 1))))
    _P_s_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q + 1, q + 1))))
    _P, _P_s = _P_R'_P_R, _P_s_R'_P_s_R
    PM, P_sM = Matrix(_P), Matrix(_P_s)

    _A = IsometricKroneckerProduct(d, rand(q + 1, q + 1))
    AM = Matrix(_A)
    _Q_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q + 1, q + 1)) + I))
    _Q = _Q_R'_Q_R
    _Q_SR = PSDMatrix(_Q_R)

    # PREDICT first
    m_p = AM * m
    _P_p_R = IsometricKroneckerProduct(d, qr([_P_R.B * _A.B'; _Q_R.B]).R |> Matrix)
    _P_p = _A * _P * _A' + _Q
    @assert _P_p ≈ _P_p_R'_P_p_R
    P_pM = Matrix(_P_p)

    # SMOOTH
    G = _P * _A' * inv(_P_p) |> Matrix
    m_smoothed = m + G * (m_s - m_p)
    P_smoothed = PM + G * (P_sM - P_pM) * G'

    x_smoothed = Gaussian(m_smoothed, P_smoothed)

    @testset "Factorization: $_FAC" for _FAC in (
        PNDE.DenseCovariance,
        PNDE.BlockDiagonalCovariance,
        PNDE.IsometricKroneckerCovariance,
    )
        FAC = _FAC{Float64}(d, q)

        P_R = PNDE.to_factorized_matrix(FAC, _P_R)
        P = P_R'P_R
        P_s_R = PNDE.to_factorized_matrix(FAC, _P_s_R)
        P_s = P_s_R'P_s_R
        P_p_R = PNDE.to_factorized_matrix(FAC, _P_p_R)
        P_p = P_p_R'P_p_R

        x_curr = Gaussian(m, P)
        x_next = Gaussian(m_s, P_s)

        A = PNDE.to_factorized_matrix(FAC, _A)
        Q_R = PNDE.to_factorized_matrix(FAC, _Q_R)
        Q = Q_R'Q_R
        Q_SR = PSDMatrix(Q_R)

        x_curr = Gaussian(m, P)
        x_next = Gaussian(m_s, P_s)

        C_DxD = PNDE.factorized_zeros(FAC, D, D)
        C_2DxD = PNDE.factorized_zeros(FAC, 2D, D)
        C_3DxD = PNDE.factorized_zeros(FAC, 3D, D)

        @testset "smooth" begin
            x_out, _ = ProbNumDiffEq.smooth(x_curr, x_next, A, Q)
            @test m_smoothed ≈ x_out.μ
            @test P_smoothed ≈ x_out.Σ
        end
        @testset "smooth with PSDMatrix" begin
            x_curr_psd = Gaussian(m, PSDMatrix(P_R)) |> copy
            x_next_psd = Gaussian(m_s, PSDMatrix(P_s_R)) |> copy
            x_out, _ = ProbNumDiffEq.smooth(x_curr_psd, x_next_psd, A, Q_SR)
            @test m_smoothed ≈ x_out.μ
            @test P_smoothed ≈ Matrix(x_out.Σ)
        end
        @testset "smooth via backward kernels" begin
            K_forward = ProbNumDiffEq.AffineNormalKernel(copy(A), copy(Q_SR))
            K_backward = ProbNumDiffEq.AffineNormalKernel(
                copy(A), copy(m_p), PSDMatrix(copy(C_2DxD)))

            x_curr = Gaussian(m, PSDMatrix(P_R)) |> copy
            x_next_pred = Gaussian(m_p, PSDMatrix(P_p_R)) |> copy
            x_next_smoothed = Gaussian(m_s, PSDMatrix(P_s_R)) |> copy

            ProbNumDiffEq.compute_backward_kernel!(
                K_backward, x_next_pred, x_curr, K_forward; C_DxD)

            G = Matrix(x_curr.Σ) * Matrix(A)' * inv(Matrix(x_next_pred.Σ))
            b = x_curr.μ - G * x_next_pred.μ
            Λ = Matrix(x_curr.Σ) - G * Matrix(x_next_pred.Σ) * G'
            @test K_backward.A ≈ G
            @test K_backward.b ≈ b
            @test Matrix(K_backward.C) ≈ Λ

            ProbNumDiffEq.marginalize_mean!(x_curr.μ, x_next_smoothed.μ, K_backward)
            ProbNumDiffEq.marginalize_cov!(
                x_curr.Σ,
                x_next_smoothed.Σ,
                K_backward;
                C_DxD,
                C_3DxD,
            )

            @test m_smoothed ≈ x_curr.μ
            @test P_smoothed ≈ Matrix(x_curr.Σ)

            @testset "test AffineNormalKernel functionality" begin
                K2 = similar(K_backward)
                @test K2 != K_backward
                @test_nowarn copy!(K2, K_backward)
                @test K2 ≈ K_backward
                @test K2 == K_backward
                @test K2.A == K_backward.A
                @test K2.b == K_backward.b
                @test K2.C == K_backward.C
            end

            @testset "smooth via backward kernels with diffusion $diffusion" for diffusion in
                                                                                 (
                rand(), rand() * I(d), Diagonal(rand(d)),
            )
                if _FAC == PNDE.IsometricKroneckerCovariance && !(diffusion isa Number)
                    continue
                end
                _diffusions =
                    diffusion isa Number ? diffusion * Ones(d) : diffusion.diag
                QM_diff = Matrix(BlocksOfDiagonals([σ² * _Q.B for σ² in _diffusions]))

                ProbNumDiffEq.compute_backward_kernel!(
                    K_backward, x_next_pred, x_curr, K_forward; C_DxD, diffusion)

                G = Matrix(x_curr.Σ) * Matrix(A)' * inv(Matrix(x_next_pred.Σ))
                b = x_curr.μ - G * x_next_pred.μ
                Λ = (I - G * AM) * Matrix(x_curr.Σ) * (I - G * AM)' + G * QM_diff * G'
                @test K_backward.A ≈ G
                @test K_backward.b ≈ b
                @test Matrix(K_backward.C) ≈ Λ
            end
        end
    end
end

@testset "Structured predict_cov and smooth ($T)" for T in (Float64, BigFloat)
    d, q = 3, 2
    D = d * (q + 1)
    uppertri(n) = Matrix(UpperTriangular(rand(T, n, n)) + I)
    dense_psd(M::PSDMatrix) = PSDMatrix(Matrix(M.R))

    m, m_s = rand(T, D), rand(T, D)
    structured_inputs = (
        IsometricKroneckerProduct => (
            PSDMatrix(IsometricKroneckerProduct(d, uppertri(q + 1))),
            PSDMatrix(IsometricKroneckerProduct(d, uppertri(q + 1))),
            IsometricKroneckerProduct(d, rand(T, q + 1, q + 1)),
            PSDMatrix(IsometricKroneckerProduct(d, uppertri(q + 1))),
        ),
        BlocksOfDiagonals => (
            PSDMatrix(BlocksOfDiagonals([uppertri(q + 1) for _ in 1:d])),
            PSDMatrix(BlocksOfDiagonals([uppertri(q + 1) for _ in 1:d])),
            BlocksOfDiagonals([rand(T, q + 1, q + 1) for _ in 1:d]),
            PSDMatrix(BlocksOfDiagonals([uppertri(q + 1) for _ in 1:d])),
        ),
    )
    @testset "$M" for (M, (P, P_s, A, Q)) in structured_inputs
        x_curr, x_next = Gaussian(m, P), Gaussian(m_s, P_s)
        x_curr_dense, x_next_dense =
            Gaussian(m, dense_psd(P)), Gaussian(m_s, dense_psd(P_s))
        A_dense, Q_dense = Matrix(A), dense_psd(Q)

        @testset "predict_cov" begin
            P_p = PNDE.predict_cov(P, A, Q)
            @test P_p.R isa M
            @test Matrix(P_p) ≈ Matrix(PNDE.predict_cov(dense_psd(P), A_dense, Q_dense))

            x_p = PNDE.predict(x_curr, A, Q)
            @test x_p.μ ≈ A_dense * m
            @test x_p.Σ.R isa M
        end

        @testset "smooth" begin
            x_s, G = PNDE.smooth(x_curr, x_next, A, Q)
            x_s_dense, G_dense = PNDE.smooth(x_curr_dense, x_next_dense, A_dense, Q_dense)
            @test x_s.Σ.R isa M
            @test G isa M
            @test eltype(x_s.μ) == T
            @test x_s.μ ≈ x_s_dense.μ
            @test Matrix(x_s.Σ) ≈ Matrix(x_s_dense.Σ)
            @test Matrix(G) ≈ G_dense
        end

        @testset "smooth with a zero-covariance next state" begin
            x_next_zero = Gaussian(m_s, PSDMatrix(zero(P_s.R)))
            x_next_zero_dense = Gaussian(m_s, PSDMatrix(zeros(T, D, D)))
            x_s, _ = PNDE.smooth(x_curr, x_next_zero, A, Q)
            x_s_dense, _ = PNDE.smooth(x_curr_dense, x_next_zero_dense, A_dense, Q_dense)
            @test x_s.Σ.R isa M
            @test x_s.μ ≈ x_s_dense.μ
            @test Matrix(x_s.Σ) ≈ Matrix(x_s_dense.Σ)
        end
    end
end

@testset "Dense output keeps the covariance structure" begin
    f!(du, u, p, t) = (du .= -u)
    prob = ODEProblem(f!, [1.0, 2.0, 3.0], (0.0, 1.0))
    @testset "$alg" for (alg, M) in (
        EK0() => IsometricKroneckerProduct,
        EK0(diffusionmodel=DynamicMVDiffusion()) => BlocksOfDiagonals,
        DiagonalEK1() => BlocksOfDiagonals,
    )
        sol = solve(prob, alg)
        t_between = (sol.t[2] + sol.t[3]) / 2
        t_after = sol.t[end] + 0.1
        interp(t) = PNDE.interpolate(
            t, sol.t, sol.x_filt, sol.x_smooth, sol.diffusions, sol.interp.cache;
            smoothed=true)
        @test interp(sol.t[2]).Σ.R isa M
        @test interp(t_between).Σ.R isa M
        @test interp(t_after).Σ.R isa M
        @test typeof(sol(t_between)) == typeof(sol(sol.t[2])) == typeof(sol(t_after))
        @test sol(t_between).Σ.R isa M
    end
end
