"""
Check the correctness of the filtering implementations vs. basic readable math code
"""

using Test
using ProbNumDiffEq
using LinearAlgebra
import ProbNumDiffEq: IsometricKroneckerProduct, BlockDiag
import ProbNumDiffEq as PNDE
import BlockDiagonals

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
            for diffusion in (rand(), rand() * Eye(d), rand() * I(d), Diagonal(rand(d)))
                if _FAC == PNDE.IsometricKroneckerCovariance &&
                    !(diffusion isa Number || diffusion isa Diagonal{<:Number,<:FillArrays.Fill})
                    continue
                end
                _diffusions = diffusion isa Number ? diffusion * Ones(d) : diffusion.diag

                QM_diff = Matrix(BlockDiagonal([σ² * _Q.B for σ² in _diffusions]))
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
                PSDMatrix(IsometricKroneckerProduct(Q_R.ldim, [Q_R.B; zero(Q_R.B)]))
            elseif Q_R isa BlockDiag
                PSDMatrix(BlockDiag([[B; zero(B)] for B in Q_R.blocks]))
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
                @test m ≈ x_out.μ
                @test P ≈ Matrix(x_out.Σ)
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
                ProbNumDiffEq.update!(
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
        end
    end
end

@testset "SMOOTH" begin
    # Setup
    d = 5
    q = 2
    D = d * (q + 1)

    m, m_s = rand(D), rand(D)
    _P_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q+1, q+1))))
    _P_s_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q+1, q+1))))
    _P, _P_s = _P_R'_P_R, _P_s_R'_P_s_R
    PM, P_sM = Matrix(_P), Matrix(_P_s)

    _A = IsometricKroneckerProduct(d, rand(q + 1, q + 1))
    AM = Matrix(_A)
    _Q_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q+1, q+1))+I))
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
        end
    end
end
