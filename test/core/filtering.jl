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

# The buffers of `update!` for an `o`-dimensional observation of a `D`-dimensional state
update_cache(FAC, D, o) = (;
    K1=PNDE.factorized_zeros(FAC, D, o),
    C_Dxd=PNDE.factorized_zeros(FAC, D, o),
    C_DxD=PNDE.factorized_zeros(FAC, D, D),
    measurement=Gaussian(zeros(o), PNDE.factorized_zeros(FAC, o, o)),
    C_dxd=PNDE.factorized_zeros(FAC, o, o),
    C_d=zeros(o),
)

@testset "UPDATE" begin
    # A scalar observation of a 5-dimensional state, which is a single block in each
    # covariance structure
    D, o = 5, 1
    m_p = rand(D)
    _P_p_R = IsometricKroneckerProduct(o, Matrix(UpperTriangular(rand(D, D))))
    _H = IsometricKroneckerProduct(o, rand(o, D))
    _R_R = IsometricKroneckerProduct(o, rand(o, o))
    z = rand(o)
    P_p, HM, RM = Matrix(_P_p_R'_P_p_R), Matrix(_H), Matrix(_R_R'_R_R)

    @testset "Factorization: $_FAC, noise: $noise" for _FAC in (
            PNDE.DenseCovariance,
            PNDE.BlockDiagonalCovariance,
            PNDE.IsometricKroneckerCovariance,
        ),
        noise in (false, true)

        FAC = _FAC{Float64}(o, D)
        P_p_R = PNDE.to_factorized_matrix(FAC, _P_p_R)
        H = PNDE.to_factorized_matrix(FAC, _H)
        R = noise ? PNDE.to_factorized_matrix(FAC, PSDMatrix(_R_R)) : nothing
        x_pred = Gaussian(m_p, PSDMatrix(P_p_R))
        obs = PNDE.LinearizedObservation(m_p, z, H, R)
        cache = update_cache(FAC, D, o)

        S = HM * P_p * HM' + (noise ? RM : zeros(o, o))
        K = P_p * HM' / S
        m, P = m_p - K * z, P_p - K * S * K'
        LL = logpdf(Gaussian(z, S), zeros(o))

        @testset "update" begin
            x_out = PNDE.update(x_pred, obs)
            @test x_out.μ ≈ m
            @test x_out.Σ ≈ P
        end

        x_out = copy(x_pred)
        res = PNDE.update!(x_out, x_pred, obs; cache)
        @testset "update!" begin
            @test x_out.μ ≈ m
            @test Matrix(x_out.Σ) ≈ P
            @test Matrix(res.S) ≈ S
            @test res.loglikelihood ≈ LL
            @test res.ztSinvz ≈ z' * (S \ z)
        end

        @testset "update_mean! and update_cov!" begin
            x_split = copy(x_pred)
            (; K, B) = PNDE.update_mean!(x_split, x_pred, obs; cache)
            @test x_split.μ == x_out.μ
            PNDE.update_cov!(x_split, x_pred, obs, K, B; cache)
            @test x_split == x_out
        end

        @testset "Linearized at a point other than the mean" begin
            # The same affine model, written at `m_lin`
            m_lin = m_p + rand(D)
            obs_lin = PNDE.LinearizedObservation(m_lin, z + HM * (m_lin - m_p), H, R)
            x_lin = copy(x_pred)
            res_lin = PNDE.update!(x_lin, x_pred, obs_lin; cache)
            @test res_lin.loglikelihood ≈ LL
            @test res_lin.ztSinvz ≈ res.ztSinvz
            @test x_lin.μ ≈ m
            @test Matrix(x_lin.Σ) ≈ P
            @test PNDE.update(x_pred, obs_lin).μ ≈ m
        end

        @testset "Zero predicted covariance" begin
            x_pred0 = Gaussian(m_p, PSDMatrix(zero(P_p_R)))
            x_out0 = copy(x_pred)
            (; loglikelihood, ztSinvz) = PNDE.update!(x_out0, x_pred0, obs; cache)
            @test x_out0 == x_pred0
            noise || @test loglikelihood == -Inf
            noise || @test ztSinvz == Inf
        end
    end
end

@testset "SolutionObservation of a ScaledSelection with a block-diagonal covariance" begin
    # Row `k` of `H = e0ᵀ ⊗ diag(m) Π` observes `m[k]` times the zeroth derivative of
    # dimension `dims[k]`. Compare `_matmul!` and `update!` with the dense `H = [M 0 ⋯ 0]`;
    # dimension 2 is not observed and keeps its prediction.
    d, q, m, dims = 3, 2, [2.0, -1.0], [3, 1]
    o, D = length(dims), d * (q + 1)
    M = PNDE.ScaledSelection(m, dims, d)
    H = PNDE.SolutionObservation(M, q)
    H_dense = [Matrix(M) zeros(o, d * q)]
    blockdiag(f, n) = BlocksOfDiagonals([f() for _ in 1:n])
    x_pred = Gaussian(
        rand(D),
        PSDMatrix(blockdiag(() -> Matrix(UpperTriangular(rand(q + 1, q + 1))), d)))
    RR = blockdiag(() -> rand(1, 1), o)
    P, R = Matrix(x_pred.Σ), Matrix(PSDMatrix(RR))

    @test PNDE._matmul!(zeros(o), H, x_pred.μ) ≈ H_dense * x_pred.μ

    # The block-wise update uses the first `o` blocks of the buffers
    FAC = PNDE.BlockDiagonalCovariance{Float64}(o, q)
    z = rand(o)
    obs = PNDE.LinearizedObservation(x_pred.μ, z, H, PSDMatrix(RR))
    x_out = Gaussian(zero(x_pred.μ), PSDMatrix(zero(x_pred.Σ.R)))
    (; loglikelihood, ztSinvz, S) =
        PNDE.update!(x_out, x_pred, obs; cache=update_cache(FAC, o * (q + 1), o))
    S_dense = H_dense * P * H_dense' + R
    K = P * H_dense' / S_dense
    @test Matrix(S) ≈ S_dense
    @test x_out.μ ≈ x_pred.μ - K * z
    @test Matrix(x_out.Σ) ≈ P - K * S_dense * K'
    @test loglikelihood ≈ logpdf(Gaussian(z, S_dense), zeros(o))
    @test ztSinvz ≈ z' * (S_dense \ z)
end

@testset "UPDATE log-likelihood with a multi-dimensional Kronecker covariance" begin
    # `S ⊗ I_d` has the log-determinant `d * logdet(S)`
    d, q1 = 3, 4
    FAC = PNDE.IsometricKroneckerCovariance{Float64}(d, q1 - 1)
    D = d * q1
    P_R = IsometricKroneckerProduct(d, Matrix(UpperTriangular(rand(q1, q1))))
    H = IsometricKroneckerProduct(d, rand(1, q1))
    m_p = rand(D)
    z = H * m_p

    x_pred = Gaussian(m_p, PSDMatrix(P_R))
    x_out = copy(x_pred)
    obs = PNDE.LinearizedObservation(m_p, z, H)
    (; loglikelihood, ztSinvz, S) =
        PNDE.update!(x_out, x_pred, obs; cache=update_cache(FAC, D, d))
    S_dense = Matrix(H) * Matrix(PSDMatrix(P_R)) * Matrix(H)'
    @test Matrix(S) ≈ S_dense
    @test loglikelihood ≈ logpdf(Gaussian(z, S_dense), zeros(d))
    @test ztSinvz ≈ z' * (S_dense \ z)
    x_ref = PNDE.update(x_pred, obs)
    @test x_out.μ ≈ x_ref.μ
    @test Matrix(x_out.Σ) ≈ x_ref.Σ
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
        C_2Dx2D = PNDE.factorized_zeros(FAC, 2D, 2D)

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
        @testset "backward kernel for an ill-conditioned prediction" begin
            # This takes the QR branch, which does not use the predicted covariance
            _R_bad = copy(_P_p_R.B)
            _R_bad[end, end] *= 1e-10
            R_bad = PNDE.to_factorized_matrix(FAC, IsometricKroneckerProduct(d, _R_bad))
            x_next_pred = Gaussian(copy(m_p), PSDMatrix(R_bad))
            x_curr = Gaussian(m, PSDMatrix(P_R)) |> copy
            K_forward = ProbNumDiffEq.AffineNormalKernel(copy(A), copy(Q_SR))
            K_backward = ProbNumDiffEq.AffineNormalKernel(
                copy(A), copy(m_p), PSDMatrix(copy(C_2DxD)))
            ProbNumDiffEq.compute_backward_kernel!(
                K_backward, x_next_pred, x_curr, K_forward; C_DxD, C_2DxD, C_2Dx2D)
            G = PM * AM' * inv(P_pM)
            @test K_backward.A ≈ G
            @test K_backward.b ≈ m - G * m_p
            @test Matrix(K_backward.C) ≈ PM - G * P_pM * G'
        end
        @testset "smooth via backward kernels" begin
            K_forward = ProbNumDiffEq.AffineNormalKernel(copy(A), copy(Q_SR))
            K_backward = ProbNumDiffEq.AffineNormalKernel(
                copy(A), copy(m_p), PSDMatrix(copy(C_2DxD)))

            x_curr = Gaussian(m, PSDMatrix(P_R)) |> copy
            x_next_pred = Gaussian(m_p, PSDMatrix(P_p_R)) |> copy
            x_next_smoothed = Gaussian(m_s, PSDMatrix(P_s_R)) |> copy

            ProbNumDiffEq.compute_backward_kernel!(
                K_backward, x_next_pred, x_curr, K_forward; C_DxD, C_2DxD, C_2Dx2D)

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
                    K_backward, x_next_pred, x_curr, K_forward;
                    C_DxD, C_2DxD, C_2Dx2D, diffusion)

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

@testset "Backward kernel from a QR decomposition ($T)" for T in (Float64, BigFloat)
    D = 8
    uppertri(n) = Matrix(UpperTriangular(rand(T, n, n)) + I)
    μ, R, A, Q_R, diffusion = rand(T, D), uppertri(D), rand(T, D, D), uppertri(D), rand(T)
    Σ = R'R
    Σ_p = A * Σ * A' + diffusion * Q_R'Q_R
    G = Σ * A' / Σ_p

    x = Gaussian(μ, PSDMatrix(R))
    x_pred = Gaussian(A * μ, PSDMatrix(zeros(T, D, D)))
    K = PNDE.AffineNormalKernel(zeros(T, D, D), zeros(T, D), PSDMatrix(zeros(T, 2D, D)))
    PNDE.qr_backward_kernel!(
        K, x_pred, x, PNDE.AffineNormalKernel(A, PSDMatrix(Q_R));
        C_2DxD=zeros(T, 2D, D), C_2Dx2D=zeros(T, 2D, 2D), diffusion)
    @test K.A ≈ G
    @test K.b ≈ μ - G * A * μ
    @test Matrix(K.C) ≈ Σ - G * Σ_p * G'
end

@testset "is_well_conditioned" begin
    M = randn(10, 4)
    @test PNDE.is_well_conditioned(qr(M).R)
    M[:, 4] .= M[:, 1] .+ 1e-10 .* randn(10)
    @test !PNDE.is_well_conditioned(qr(M).R)
    # the check does not depend on the scaling of the columns
    @test !PNDE.is_well_conditioned(qr(M * Diagonal([1e10, 1, 1, 1e-10])).R)
end
