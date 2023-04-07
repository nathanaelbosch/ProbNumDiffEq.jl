"""
Check the correctness of the filtering implementations vs. basic readable math code
"""

using Test
using ProbNumDiffEq
using LinearAlgebra

@testset "PREDICT" begin
    # Setup
    d = 5
    m = rand(d)
    P_R = Matrix(UpperTriangular(rand(d, d)))
    P = P_R'P_R

    A = rand(d, d)
    Q_R = Matrix(UpperTriangular(rand(d, d)))
    Q = Q_R'Q_R

    # PREDICT
    m_p = A * m
    P_p = A * P * A' + Q

    x_curr = Gaussian(m, P)
    x_out = copy(x_curr)

    @testset "predict" begin
        x_out = ProbNumDiffEq.predict(x_curr, A, Q)
        @test m_p == x_out.μ
        @test P_p == x_out.Σ
    end

    @testset "predict! with PSDMatrix" begin
        x_curr = Gaussian(m, PSDMatrix(P_R))
        x_out = copy(x_curr)
        Q_SR = PSDMatrix(Q_R)
        ProbNumDiffEq.predict!(x_out, x_curr, A, Q_SR, zeros(d, d), zeros(2d, d))
        @test m_p == x_out.μ
        @test P_p ≈ Matrix(x_out.Σ)
    end

    @testset "predict! with zero diffusion" begin
        x_curr = Gaussian(m, PSDMatrix(P_R))
        x_out = copy(x_curr)
        Q_SR = PSDMatrix(Q_R)
        ProbNumDiffEq.predict!(x_out, x_curr, A, Q_SR, zeros(d, d), zeros(2d, d), 0)
        @test m_p == x_out.μ
        @test Matrix(x_out.Σ) ≈ Matrix(X_A_Xt(x_curr.Σ, A))
    end

    @testset "predict with kernel and marginalize!" begin
        x_curr = Gaussian(m, PSDMatrix(P_R))
        x_out = copy(x_curr)
        Q_SR = PSDMatrix([Q_R; zero(Q_R)]) # marginalize! needs tall square-roots
        K = ProbNumDiffEq.AffineNormalKernel(A, Q_SR)
        ProbNumDiffEq.marginalize!(x_out, x_curr, K; C_DxD=zeros(d, d), C_3DxD=zeros(3d, d))
        @test m_p == x_out.μ
        @test P_p ≈ Matrix(x_out.Σ)
    end
end

@testset "UPDATE" begin
    # Setup
    d = 5
    m_p = rand(d)
    P_p_R = Matrix(UpperTriangular(rand(d, d)))
    P_p = P_p_R'P_p_R

    # Measure
    o = 3
    H = rand(o, d)
    R = zeros(o, o)

    z_data = zeros(o)
    z = H * m_p
    S = Symmetric(H * P_p * H' + R)

    # UPDATE
    S_inv = inv(S)
    K = P_p * H' * S_inv
    m = m_p + K * (z_data .- z)
    P = P_p - K * S * K'

    x_pred = Gaussian(m_p, P_p)
    x_out = copy(x_pred)
    measurement = Gaussian(z, S)

    @testset "update" begin
        x_out = ProbNumDiffEq.update(x_pred, measurement, H)
        @test m ≈ x_out.μ
        @test P ≈ x_out.Σ
    end

    @testset "update!" begin
        # @testset "Array" begin
        #     K_cache = copy(K)
        #     K2_cache = copy(K)
        #     M_cache = zeros(d, d)
        #     O_cache = zeros(o, o)
        #     ProbNumDiffEq.update!(
        #         x_out,
        #         x_pred,
        #         measurement,
        #         H,
        #         K_cache,
        #         K2_cache,
        #         M_cache,
        #         O_cache,
        #     )
        #     @test m ≈ x_out.μ
        #     @test P ≈ Matrix(x_out.Σ)
        # end
        @testset "PSDMatrix" begin
            K_cache = copy(K)
            K2_cache = copy(K)
            M_cache = zeros(d, d)
            S = measurement.Σ
            SR = cholesky(S).U |> Matrix
            msmnt = Gaussian(measurement.μ, PSDMatrix(SR))
            O_cache = zeros(o, o)
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
            )
            @test m ≈ x_out.μ
            @test P ≈ Matrix(x_out.Σ)
        end
        @testset "Zero predicted covariance" begin
            K_cache = copy(K)
            K2_cache = copy(K)
            M_cache = zeros(d, d)
            S = measurement.Σ
            SR = cholesky(S).U
            msmnt = Gaussian(measurement.μ, PSDMatrix(SR))
            O_cache = zeros(o, o)
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
            )
            @test x_out == x_pred
        end
        @testset "Positive semi-definite measurement cov" begin
            # different measurement matrix to make sure the measurement cov is not posdef
            H = [rand(o - 1, d); zeros(1, d)]

            z_data = zeros(o)
            z = H * m_p
            S = PSDMatrix(P_p_R * H')

            K_cache = copy(K)
            K2_cache = copy(K)
            M_cache = zeros(d, d)
            msmnt = Gaussian(measurement.μ, PSDMatrix(S.R))
            O_cache = zeros(o, o)
            x_pred = Gaussian(x_pred.μ, PSDMatrix(P_p_R))
            x_out = copy(x_pred)

            warnmsg = "Can't compute the update step with cholesky; using qr instead"
            @test_logs (:warn, warnmsg) ProbNumDiffEq.update!(
                x_out,
                x_pred,
                msmnt,
                H,
                K_cache,
                K2_cache,
                M_cache,
                O_cache,
            )
        end
    end
end

@testset "SMOOTH" begin
    # Setup
    d = 5
    m, m_s = rand(d), rand(d)
    P_R, P_s_R = Matrix(UpperTriangular(rand(d, d))), Matrix(UpperTriangular(rand(d, d)))
    P, P_s = P_R'P_R, P_s_R'P_s_R

    A = rand(d, d)
    Q_R = Matrix(UpperTriangular(rand(d, d)))
    Q = Q_R'Q_R
    Q_SR = PSDMatrix(Q_R)

    # PREDICT first
    m_p = A * m
    P_p_R = qr([P_R * A'; Q_R]).R |> Matrix
    P_p = A * P * A' + Q
    @assert P_p ≈ P_p_R'P_p_R

    # SMOOTH
    G = P * A' * inv(P_p)
    m_smoothed = m + G * (m_s - m_p)
    P_smoothed = P + G * (P_s - P_p) * G'

    x_curr = Gaussian(m, P)
    x_next = Gaussian(m_s, P_s)
    x_smoothed = Gaussian(m_smoothed, P_smoothed)

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
    @testset "smooth!" begin
        x_curr_psd = Gaussian(m, PSDMatrix(P_R)) |> copy
        x_next_psd = Gaussian(m_s, PSDMatrix(P_s_R)) |> copy
        cache = (
            x_pred=copy(x_curr_psd),
            G1=zeros(d, d),
            C_DxD=zeros(d, d),
            C_2DxD=zeros(2d, d),
            C_3DxD=zeros(3d, d),
        )
        ProbNumDiffEq.smooth!(
            x_curr_psd,
            x_next_psd,
            A,
            Q_SR,
            cache,
        )
        @test m_smoothed ≈ x_curr_psd.μ
        @test P_smoothed ≈ Matrix(x_curr_psd.Σ)
    end

    @testset "smooth via backward kernels" begin
        K_forward = ProbNumDiffEq.AffineNormalKernel(copy(A), copy(Q_SR))
        K_backward = ProbNumDiffEq.AffineNormalKernel(
            similar(A), similar(m_p), PSDMatrix(zeros(2d, d)))

        x_curr = Gaussian(m, PSDMatrix(P_R)) |> copy
        x_next_pred = Gaussian(m_p, PSDMatrix(P_p_R)) |> copy
        x_next_smoothed = Gaussian(m_s, PSDMatrix(P_s_R)) |> copy

        C_DxD = zeros(d, d)
        ProbNumDiffEq.compute_backward_kernel!(
            K_backward, x_next_pred, x_curr, K_forward; C_DxD)

        G = Matrix(x_curr.Σ) * A' * inv(Matrix(x_next_pred.Σ))
        b = x_curr.μ - G * x_next_pred.μ
        Λ = Matrix(x_curr.Σ) - G * Matrix(x_next_pred.Σ) * G'
        @test K_backward.A ≈ G
        @test K_backward.b ≈ b
        @test Matrix(K_backward.C) ≈ Λ

        C_3DxD = zeros(3d, d)
        ProbNumDiffEq.marginalize_mean!(x_curr, x_next_smoothed, K_backward)
        ProbNumDiffEq.marginalize_cov!(x_curr, x_next_smoothed, K_backward; C_DxD, C_3DxD)

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
