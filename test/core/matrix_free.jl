using Test
using ProbNumDiffEq
using LinearAlgebra
using Random
import ProbNumDiffEq as PNDE
using ProbNumDiffEq: BlocksOfDiagonals, LinearizedObservation

# A vector field whose Jacobian J = A + Diagonal(1 .- 2u) is dense and not symmetric
const d, q = 10, 3
const D = d * (q + 1)
const A = randn(Xoshiro(1), d, d)
f!(du, u, p, t) = (mul!(du, A, u); du .+= u .* (1 .- u); du)
jvp!(Jv, v, u, p, t) = (mul!(Jv, A, v); Jv .+= (1 .- 2u) .* v; Jv)
vjp!(Jᵀv, v, u, p, t) = (mul!(Jᵀv, A', v); Jᵀv .+= (1 .- 2u) .* v; Jᵀv)
diagonal_jacobian!(out, u, p, t) = (out .= diag(A) .+ 1 .- 2u)
const f = ODEFunction(f!; jvp=jvp!, vjp=vjp!)

rng = Xoshiro(2)
u = randn(rng, d)
J = A + Diagonal(1 .- 2u)
E0, E1 = PNDE.projection(d, q)(0), PNDE.projection(d, q)(1)
H_dense = E1 - J * E0
R = BlocksOfDiagonals([Matrix(UpperTriangular(randn(rng, q + 1, q + 1))) for _ in 1:d])
Σ = Matrix(R)' * Matrix(R)
S_dense = H_dense * Σ * H_dense'

function jacobian_products(; rtol=1e-12, kwargs...)
    linearization = MatrixFreeJacobian(; kwargs...)
    C = PNDE.BlockDiagonalCovariance{Float64}(d, q)
    cache = PNDE.make_linearization_cache(linearization, C, u, f)
    copy!(cache.u, u)
    isnothing(cache.diagonal_jacobian) ||
        linearization.diagonal_jacobian(cache.diagonal_jacobian, u, nothing, 0.0)
    return PNDE.JacobianProducts(f, cache.u, nothing, 0.0, Xoshiro(3), rtol, cache)
end
observed_covariance(H, R) =
    PNDE.ObservedCovariance(H, PNDE.first_two_columns!(H.cache.Σcols, R))

@testset "Products with H Σ Hᵀ and Σ Hᵀ" begin
    S = observed_covariance(jacobian_products(), R)
    w, W = randn(rng, d), randn(rng, d, 3)
    @test mul!(zeros(d), S, w) ≈ S_dense * w
    @test mul!(zeros(d, 3), S, W) ≈ S_dense * W
    @test PNDE.gain_times!(zeros(D), S, w) ≈ Σ * H_dense' * w
end

@testset "Diagonals for the Jacobi preconditioners" begin
    H = jacobian_products(; diagonal_jacobian=diagonal_jacobian!)
    H_diagonal = E1 - Diagonal(J) * E0
    @test PNDE.diagonal_jacobian_diag!(zeros(d), observed_covariance(H, R)) ≈
          diag(H_diagonal * Σ * H_diagonal')
    @test PNDE.estimate_observed_diagonal!(zeros(d), R, H, 20_000) ≈ diag(S_dense) rtol =
        0.05
end

@testset "Mean update, linearized at $point" for point in ("the mean", "another point")
    H = jacobian_products()
    μ = randn(rng, D)
    x_pred = Gaussian(μ, PSDMatrix(R))
    z = randn(rng, d)
    m = point == "the mean" ? μ : μ + randn(rng, D)
    obs = LinearizedObservation(m, z + H_dense * (m - μ), H)
    x_out = copy(x_pred)
    upd = PNDE.update_mean!(x_out, x_pred, obs; cache=nothing)
    @test x_out.μ ≈ μ - Σ * H_dense' * (S_dense \ z)
    @test upd.ztSinvz ≈ z' * (S_dense \ z)
    @test isnan(upd.loglikelihood)
end

@testset "Probe systems solved $how" for (how, nprobes) in
                                         (("jointly by block-GMRES", 5), ("by CG", 8))
    H = jacobian_products(; nprobes)
    c = H.cache
    @test isnothing(c.block_gmres) == (how == "by CG")
    c.P .= diag(S_dense)
    randn!(rng, c.B)
    @test PNDE.solve_probes!(c, observed_covariance(H, R), 1e-12) ≈ S_dense \ c.B
    @test !c.failed[]
end

@testset "Covariance update" begin
    H = jacobian_products(; nprobes=4000)
    x_pred = Gaussian(randn(rng, D), PSDMatrix(R))
    obs = LinearizedObservation(x_pred.μ, randn(rng, d), H)
    x_out = copy(x_pred)
    upd = PNDE.update_mean!(x_out, x_pred, obs; cache=nothing)
    PNDE.update_cov!(x_out, x_pred, obs, upd.K, upd.B; cache=nothing)
    # The diagonal blocks of the dense posterior, up to the Monte Carlo error
    Σ_posterior = Σ - Σ * H_dense' * (S_dense \ (H_dense * Σ))
    for i in 1:d
        block = i:d:D
        @test Matrix(x_out.Σ)[block, block] ≈ Σ_posterior[block, block] rtol = 0.15
    end
end

RQ = BlocksOfDiagonals([Matrix(UpperTriangular(randn(rng, q + 1, q + 1))) for _ in 1:d])
function observed_process_noise(H)
    H.cache.has_HQH_diag[] = false
    return PNDE.ObservedProcessNoise(
        PNDE.ObservedCovariance(H, PNDE.first_two_columns!(H.cache.Qcols, RQ)), RQ)
end

@testset "Calibration" begin
    HQH = observed_process_noise(jacobian_products())
    HQH_dense = H_dense * Matrix(RQ)' * Matrix(RQ) * H_dense'
    z = randn(rng, d)
    # CG approaches zᵀ S⁻¹ z from below; its tolerance for the calibration is 0.1
    exact = z' * (HQH_dense \ z)
    @test PNDE.invquad(z, HQH) ≈ exact rtol = 0.15
    @test PNDE.invquad(z, HQH) <= exact
end

@testset "The diagonal of H Q Hᵀ is estimated once per step, when first needed" begin
    # Without a diagonal Jacobian, the calibration needs it for its preconditioner
    H = jacobian_products()
    HQH = observed_process_noise(H)
    @test !H.cache.has_HQH_diag[]
    PNDE.invquad(randn(rng, d), HQH)
    @test H.cache.has_HQH_diag[]
    estimate = copy(H.cache.HQH_diag)
    @test PNDE.diag!(zeros(d), HQH) == estimate
    # With one, only the error estimate needs it
    H = jacobian_products(; diagonal_jacobian=diagonal_jacobian!)
    HQH = observed_process_noise(H)
    PNDE.invquad(randn(rng, d), HQH)
    @test !H.cache.has_HQH_diag[]
    PNDE.diag!(zeros(d), HQH)
    @test H.cache.has_HQH_diag[]
end

@testset "Linearization" begin
    prob = ODEProblem(f, randn(rng, d), (0.0, 1.0))
    integ = init(prob, MatfreeEK1(order=q))
    m = randn(rng, D)
    obs = PNDE.linearize_ode_observation!(integ.alg.linearization, integ, m)
    @test obs.m === m
    @test obs.r ≈ E1 * m - f!(zeros(d), E0 * m, nothing, integ.t + integ.dt)
    v = randn(rng, d)
    J_m = A + Diagonal(1 .- 2 .* (E0 * m))
    @test PNDE.jvp!(zeros(d), obs.H, v) ≈ J_m * v
    @test PNDE.vjp!(zeros(d), obs.H, v) ≈ J_m' * v
    @test obs.H.rtol == 1e-3
end
