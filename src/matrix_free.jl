const DIAGONAL_PROBES = 32
const CALIBRATION_RTOL = 0.1
const MAX_CG_ITERATIONS = 10_000

"""
    throw_if_unsupported(::MatrixFreeJacobian, alg, f, u)

Throw an `ArgumentError` if the `MatfreeEK1` does not support the settings of `alg`, the ODE
function `f`, or the state `u`. It does not support second-order ODEs, mass matrices,
observation noise, or states that are not floating-point numbers. Of the multivariate
diffusion models, it only supports the `FixedMVDiffusion` with `calibrate=false`, since the
others estimate one diffusion per dimension, which assumes that `H` does not couple the
dimensions. It needs `jvp` and `vjp` in the ODE function, and at least as many probes as
there are derivatives in the state.
"""
function throw_if_unsupported(linearization::MatrixFreeJacobian, alg, f, u)
    unsupported(what) = ArgumentError("The `MatfreeEK1` does not support $what.")
    if f isa DynamicalODEFunction
        throw(unsupported("second-order ODEs"))
    elseif f.mass_matrix !== I
        throw(unsupported("mass matrices"))
    elseif !isnothing(alg.pn_observation_noise)
        throw(unsupported("observation noise"))
    elseif !(
        alg.diffusionmodel isa Union{DynamicDiffusion,FixedDiffusion} ||
        (alg.diffusionmodel isa FixedMVDiffusion && !alg.diffusionmodel.calibrate)
    )
        throw(
            unsupported(
                "multivariate diffusion models, except the `FixedMVDiffusion` with " *
                "`calibrate=false`"),
        )
    elseif !(eltype(u) <: AbstractFloat)
        # Krylov.jl needs floating-point numbers
        throw(unsupported("states that are not floating-point numbers"))
    elseif isnothing(f.jvp) || isnothing(f.vjp)
        throw(
            ArgumentError(
                "The `MatfreeEK1` needs the Jacobian-vector product " *
                "`jvp(Jv, v, u, p, t)` and the vector-Jacobian product " *
                "`vjp(Jᵀv, v, u, p, t)` of the ODE function. Pass them as " *
                "`ODEFunction(f; jvp, vjp)`."),
        )
    end
    q = num_derivatives(alg.prior)
    m = nprobes(linearization, q)
    if m < q + 1
        throw(
            ArgumentError(
                "`nprobes` must be at least the number of derivatives in the state, " *
                "$(q + 1), but it is $m."),
        )
    end
    return nothing
end

nprobes(linearization::MatrixFreeJacobian, q) =
    isnothing(linearization.nprobes) ? 2(q + 1) : linearization.nprobes

function make_linearization_cache(
    linearization::MatrixFreeJacobian, C::BlockDiagonalCovariance{T}, u, f,
) where {T}
    d, q = C.d, C.q
    D = d * (q + 1)
    m = nprobes(linearization, q)
    dvec() = zeros(T, d)
    observed_covariance() =
        (; ΣEᵀ=(zeros(T, D), zeros(T, D)), preconditioner=dvec(), diagonal=dvec())
    return (;
        r=dvec(), u=dvec(),
        diagonal_jacobian=isnothing(linearization.diagonal_jacobian) ? nothing : dvec(),
        v=dvec(), Jv=dvec(), a=dvec(), e0=dvec(), z=dvec(),
        HΣHᵀ=observed_covariance(), HQHᵀ=observed_covariance(),
        ξ=zeros(T, 2d), L=(dvec(), dvec(), dvec()),
        Ξ=zeros(T, D, m), G=zeros(T, D, m), HG=zeros(T, d, m),
        Y=zeros(T, m, q + 1), M=zeros(T, q + 1, q + 1), gtmp=zeros(T, D),
        cg=CgWorkspace(d, d, Vector{T}), W=zeros(T, d, m),
    )
end

"""
    AbstractMatrixFreeOperator

A minimal interface for matrix-free linear operators.

Operators are expected to implement a matrix-vector `mul!`, ie
```
mul!(y::AbstractVector, H::MatrixFreeOperator, x::AbstractVector)
```
For matrix-matrix multiplication this is then by default applied column by column.
"""
abstract type AbstractMatrixFreeOperator end

function mul!(Y::AbstractMatrix, A::AbstractMatrixFreeOperator, X::AbstractMatrix)
    for j in axes(X, 2)
        mul!(view(Y, :, j), A, view(X, :, j))
    end
    return Y
end

"""
    MatrixFreeObservationOperator(integ, cache)

The observation matrix ``H = E_1 - J E_0`` of the ODE, as an operator.

``E_0`` and ``E_1`` select the solution and its first derivative from the state.
``J`` is the Jacobian of the ODE at the solution `cache.u`, and it is never formed.
The main cost of ``H x = E_1 x - J E_0 x`` is thus one Jacobian-vector product (`apply_jvp!`).
The cross-covariance ``Σ H^T`` instead needs products with ``J^T`` (`apply_vjp!`).

The integrator `integ` provides the ODE function, its parameters and the time. It also
provides the random number generator for the probes, the tolerance of the linear solves,
and the statistics that count the products and the solves.
`cache` holds the buffers of the linearization.
"""
struct MatrixFreeObservationOperator{I,C} <: AbstractMatrixFreeOperator
    integ::I
    cache::C
end

const MatrixFreeLinearizedObservation =
    LinearizedObservation{<:Any,<:Any,<:MatrixFreeObservationOperator}

# The `rng` that `solve` takes in recent versions of OrdinaryDiffEq. Older ones have none.
@static if isdefined(SciMLBase, :has_rng)
    integrator_rng(integ) =
        SciMLBase.has_rng(integ) ? SciMLBase.get_rng(integ) : Random.default_rng()
else
    integrator_rng(integ) = Random.default_rng()
end

function linearize_ode_observation!(
    linearization::MatrixFreeJacobian, integ, m::AbstractVector,
)
    (; d, measurement_model, linearization_cache, R) = integ.cache
    c = linearization_cache
    p, t = integ.p, integ.t + integ.dt
    measurement_model(c.r, m, p, t)
    integ.stats.nf += 1
    copy!(c.u, view(m, 1:d))
    if !isnothing(c.diagonal_jacobian)
        linearization.diagonal_jacobian(c.diagonal_jacobian, c.u, p, t)
    end
    return LinearizedObservation(m, c.r, MatrixFreeObservationOperator(integ, c), R)
end

"""
    apply_jvp!(H::MatrixFreeObservationOperator, v)

For a given ``H = E_1 - J E_0`` with `H::MatrixFreeObservationOperator`,
compute ``J v`` as a jacobian-vector product (JVP) by calling the `jvp` of
the ODE function.

If `v` is not a `Vector`, it is first copied into one, because `f.jvp` may require it.
"""
function apply_jvp!(H::MatrixFreeObservationOperator, v)
    (; f, p, t, dt) = H.integ
    Jv = H.cache.Jv
    v = v isa Vector ? v : copy!(H.cache.v, v)
    f.jvp(Jv, v, H.cache.u, p, t + dt)
    H.integ.sol.pnstats.njvp += 1
    return Jv
end

"""
    apply_vjp!(H::MatrixFreeObservationOperator, w)

For a given ``H = E_1 - J E_0`` with `H::MatrixFreeObservationOperator`,
compute ``J^T v`` as a vector-jacobian product (VJP) by calling the `vjp` of
the ODE function.

If `v` is not a `Vector`, it is first copied into one, because `f.vjp` may require it.
"""
function apply_vjp!(H::MatrixFreeObservationOperator, w)
    (; f, p, t, dt) = H.integ
    Jᵀw = H.cache.a
    w = w isa Vector ? w : copy!(H.cache.v, w)
    f.vjp(Jᵀw, w, H.cache.u, p, t + dt)
    H.integ.sol.pnstats.nvjp += 1
    return Jᵀw
end

function mul!(y::AbstractVector, H::MatrixFreeObservationOperator, x::AbstractVector)
    d = length(y)
    E₀x, E₁x = view(x, 1:d), view(x, (d+1):2d)
    JE₀x = apply_jvp!(H, E₀x)
    @.. y = E₁x - JE₀x
    return y
end

"""
    MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, buffers)

The cross-covariance ``Σ H^T`` between the state and the observation, as an operator.

Since ``H = E_1 - J E_0``, we have ``Σ H^T w = Σ E_1^T w - Σ E_0^T J^T w``.
The constructor pre-computes ``Σ E_0^T`` and ``Σ E_1^T`` into the passed `buffers`.
Since ``Σ`` is block-diagonal, ``Σ E_0^T`` and ``Σ E_1^T`` each have exactly one
nonzero entry per row and we can store them as vectors.
"""
struct MatrixFreeCrossCovariance{HT,V} <: AbstractMatrixFreeOperator
    H::HT
    ΣE₀ᵀ::V
    ΣE₁ᵀ::V
    function MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, (ΣE₀ᵀ, ΣE₁ᵀ))
        d = nblocks(R)
        for (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
            ΣE₀ᵀ[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 1))
            ΣE₁ᵀ[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 2))
        end
        return new{typeof(H),typeof(ΣE₀ᵀ)}(H, ΣE₀ᵀ, ΣE₁ᵀ)
    end
end

function mul!(out::AbstractVector, ΣHᵀ::MatrixFreeCrossCovariance, w::AbstractVector)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = ΣHᵀ
    d = length(w)
    Jᵀw = apply_vjp!(H, w)
    for k in 0:(length(out)÷d-1)
        idx = (k*d+1):((k+1)*d)
        @views out[idx] .= ΣE₁ᵀ[idx] .* w .- ΣE₀ᵀ[idx] .* Jᵀw
    end
    return out
end

"""
    MatrixFreeObservedCovariance(H, R::BlocksOfDiagonals, buffers; needs_diagonal)

The covariance ``H Σ H^T`` of the observation, as an operator.

The main use of this operator is for matrix-free solutions of linear systems with
Krylov.jl. In the update ``Σ`` is the predicted covariance, but for error-estimation
and calibration we can also use this operator with ``Σ = Q(h)``.

To be efficient, the linear solves need a preconditioner.
The constructor therefore automatically computes a Jacobi preconditioner as an
approximation of the diagonal, in one of two ways:
if a user explicitly passed a function for the diagonal of the ODE jacobian,
we use that to compute an approximated diagonal;
otherwise we estimate it stochastically.

As the local error estimate needs the stochastic diagonal, one can also compute both
by setting `needs_diagonal=true`.
"""
struct MatrixFreeObservedCovariance{CT,RT,V} <: AbstractMatrixFreeOperator
    ΣHᵀ::CT
    R::RT
    preconditioner::V
    diagonal::V
    function MatrixFreeObservedCovariance(
        H, R::BlocksOfDiagonals, buffers; needs_diagonal::Bool,
    )
        ΣHᵀ = MatrixFreeCrossCovariance(H, R, buffers.ΣEᵀ)
        (; preconditioner, diagonal) = buffers
        S = new{typeof(ΣHᵀ),typeof(R),typeof(diagonal)}(ΣHᵀ, R, preconditioner, diagonal)
        J_D = H.cache.diagonal_jacobian
        if needs_diagonal || isnothing(J_D)
            estimate_diagonal!(diagonal, S, DIAGONAL_PROBES)
        end
        if isnothing(J_D)
            copy!(preconditioner, diagonal)
        else
            estimate_diagonal_with_diagonal_jacobian!(preconditioner, S, J_D)
        end
        # A Jacobi preconditioner needs positive entries
        T = eltype(preconditioner)
        lower = max(floatmin(T), eps(T) * maximum(preconditioner))
        @.. preconditioner = max(preconditioner, lower)
        return S
    end
end

# Krylov.jl reads the size and the element type of the operators it solves with
size(S::MatrixFreeObservedCovariance) = (length(S.diagonal), length(S.diagonal))
size(S::MatrixFreeObservedCovariance, i::Integer) = length(S.diagonal)
eltype(S::MatrixFreeObservedCovariance) = eltype(S.diagonal)

# H Σ Hᵀ w = E₁ Σ Hᵀ w - J E₀ Σ Hᵀ w. Of Σ Hᵀ w, only the two derivatives that H reads are
# formed.
function mul!(y::AbstractVector, S::MatrixFreeObservedCovariance, w::AbstractVector)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = S.ΣHᵀ
    d = length(w)
    Jᵀw = apply_vjp!(H, w)
    E₀ΣHᵀw = H.cache.e0
    @views E₀ΣHᵀw .= ΣE₁ᵀ[1:d] .* w .- ΣE₀ᵀ[1:d] .* Jᵀw
    @views y .= ΣE₁ᵀ[(d+1):2d] .* w .- ΣE₀ᵀ[(d+1):2d] .* Jᵀw
    y .-= apply_jvp!(H, E₀ΣHᵀw)
    return y
end

"""
    estimate_diagonal!(out, S::MatrixFreeObservedCovariance, n)

Estimate the diagonal of ``S = H Σ H^T`` from `n` probes, with an estimator of the
Hutchinson type. The diagonal of ``S`` is the expectation of ``(H g)^2`` for samples ``g``
with the covariance ``Σ``, and each sample costs one Jacobian-vector product. ``H`` only
reads ``E_0 g`` and ``E_1 g``, so only these are sampled. Since ``Σ`` is block-diagonal,
the pairs ``(u_i, u_i')`` of different dimensions are independent. So each pair is drawn
from its own ``2 × 2`` covariance in ``Σ``, from two entries of a probe.
"""
function estimate_diagonal!(out, S::MatrixFreeObservedCovariance, n)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = S.ΣHᵀ
    c = H.cache
    d = length(out)
    # The Cholesky factor [l₀₀ 0; l₁₀ l₁₁] of the covariance of (uᵢ, uᵢ')
    Σ₀₀, Σ₀₁, Σ₁₁ = view(ΣE₀ᵀ, 1:d), view(ΣE₁ᵀ, 1:d), view(ΣE₁ᵀ, (d+1):2d)
    l₀₀, l₁₀, l₁₁ = c.L
    @.. l₀₀ = sqrt(Σ₀₀)
    @.. l₁₀ = ifelse(l₀₀ > 0, Σ₀₁ / l₀₀, zero(l₀₀))
    @.. l₁₁ = sqrt(max(Σ₁₁ - l₁₀^2, zero(l₁₀)))
    ξ₀, ξ₁ = view(c.ξ, 1:d), view(c.ξ, (d+1):2d)
    E₀g, E₁g = c.e0, c.a
    rng = integrator_rng(H.integ)
    fill!(out, 0)
    for _ in 1:n
        Random.randn!(rng, c.ξ)
        @.. E₀g = l₀₀ * ξ₀
        @.. E₁g = l₁₀ * ξ₀ + l₁₁ * ξ₁
        JE₀g = apply_jvp!(H, E₀g)
        @.. out = out + (E₁g - JE₀g)^2
    end
    @.. out = out / n
    return out
end

"""
    estimate_diagonal_with_diagonal_jacobian!(out, S::MatrixFreeObservedCovariance, J_D)

Estimate the diagonal of ``S = H Σ H^T`` by the diagonal of ``H_D Σ H_D^T``, with
``H_D = E_1 - J_D E_0`` for the diagonal ``J_D`` of the Jacobian. Row ``i`` of ``H_D`` only
reads ``u_i`` and ``u_i'``, so its entry only needs the ``i``-th block of the square root
``R`` of ``Σ``.
"""
function estimate_diagonal_with_diagonal_jacobian!(
    out, S::MatrixFreeObservedCovariance, J_D,
)
    for (i, Ri) in enumerate(blocks(S.R))
        # (H_D Σ H_Dᵀ)ᵢᵢ = ‖R H_Dᵀ eᵢ‖² = ‖r₁ - J_Dᵢ r₀‖², with the columns r₀ and r₁ of Rᵢ
        out[i] = sum(l -> (Ri[l, 2] - J_D[i] * Ri[l, 1])^2, axes(Ri, 1))
    end
    return out
end

function krylov_rtol(H::MatrixFreeObservationOperator, part)
    T = eltype(H.cache.u)
    part === :calibration && return T(CALIBRATION_RTOL)
    # CG cannot reach the residual of zero that a `reltol` of zero asks for
    return clamp(T(minimum(H.integ.opts.reltol)), sqrt(eps(T)), T(1e-2))
end

struct LinearSolveFailure <: Exception end

"""
    krylov_solve!(S::MatrixFreeObservedCovariance, b, part)

Solve ``S x = b`` by CG with the Jacobi preconditioner of `S`, and return ``x``. A matrix
`b` is solved column by column. `part` is `:mean`, `:covariance` or `:calibration`. It sets
the tolerance, and the solves count in its statistics. If `b` or the preconditioner is not
finite, or if a solve does not converge, it throws a `LinearSolveFailure`.
"""
function krylov_solve!(
    S::MatrixFreeObservedCovariance, b::AbstractVector, part; itmax=MAX_CG_ITERATIONS,
)
    throw_if_not_finite(S, b)
    H = S.ΣHᵀ.H
    workspace = H.cache.cg
    rtol = krylov_rtol(H, part)
    M = Diagonal(S.preconditioner)
    cg!(workspace, S, b; M, ldiv=true, atol=zero(rtol), rtol, itmax)
    count_solve!(H, part, Krylov.iteration_count(workspace))
    Krylov.issolved(workspace) || throw(LinearSolveFailure())
    return Krylov.solution(workspace)
end
function krylov_solve!(S::MatrixFreeObservedCovariance, B::AbstractMatrix, part)
    W = S.ΣHᵀ.H.cache.W
    for j in axes(B, 2)
        copy!(view(W, :, j), krylov_solve!(S, view(B, :, j), part))
    end
    return W
end

# Krylov.jl throws its own errors for NaN, which would end the solve instead of rejecting the
# step, and an infinite preconditioner makes it skip the solve
function throw_if_not_finite(S::MatrixFreeObservedCovariance, b)
    if !(all(isfinite, b) && all(isfinite, S.preconditioner))
        throw(LinearSolveFailure())
    end
    return nothing
end

function count_solve!(H::MatrixFreeObservationOperator, part, niter)
    stats = H.integ.sol.pnstats
    H.integ.stats.nsolve += 1
    if part === :mean
        stats.nsolve_mean += 1
        stats.niter_mean += niter
    elseif part === :covariance
        stats.nsolve_covariance += 1
        stats.niter_covariance += niter
    else
        @assert part === :calibration
        stats.nsolve_calibration += 1
        stats.niter_calibration += niter
    end
    return nothing
end

"""
    MatrixFreeGain(S::MatrixFreeObservedCovariance)

The Kalman gain ``K = Σ H^T S^{-1}``, from the cross-covariance ``Σ H^T`` and the observed
covariance ``S``. `update_mean!` returns the gain for `update_cov!`.
"""
struct MatrixFreeGain{CT,ST}
    ΣHᵀ::CT
    S::ST
end
MatrixFreeGain(S::MatrixFreeObservedCovariance) = MatrixFreeGain(S.ΣHᵀ, S)

"""
    update_mean!(x_out, x_pred, obs::MatrixFreeLinearizedObservation; cache)

The mean of the update on a matrix-free observation. It solves ``S w = z`` for
``S = H Σ H^T`` by CG, and sets ``μ^F = μ - Σ H^T w``. It returns the gain
``K = Σ H^T S^{-1}`` for `update_cov!`, and the log-likelihood as `NaN`.
"""
function update_mean!(x_out, x_pred, obs::MatrixFreeLinearizedObservation; cache)
    (; m, H) = obs
    c = H.cache
    S = MatrixFreeObservedCovariance(H, x_pred.Σ.R, c.HΣHᵀ; needs_diagonal=false)
    z = if m === x_pred.μ
        obs.r
    else
        # z = r + H (μ - m), with m - μ in x_out.μ
        Δ = x_out.μ .= m .- x_pred.μ
        HΔ = mul!(c.z, H, Δ)
        @.. c.z = obs.r - HΔ
    end
    w = krylov_solve!(S, z, :mean)
    ztSinvz = dot(z, w)
    mul!(x_out.μ, S.ΣHᵀ, w)
    @.. x_out.μ = x_pred.μ - x_out.μ
    K = MatrixFreeGain(S)
    return (; loglikelihood=oftype(ztSinvz, NaN), ztSinvz, S, z, K, B=nothing)
end

"""
    update_cov!(x_out, x_pred, obs::MatrixFreeLinearizedObservation, K, B; cache)

The covariance of the update on a matrix-free observation. It approximates the posterior
covariance by its block-diagonal part, which it estimates from samples. The exact posterior
covariance is not block-diagonal, because ``J`` couples the blocks. The samples are
``g - K H g`` for samples ``g`` with the covariance ``Σ``, so they have the covariance
``Σ^F``.
"""
function update_cov!(
    x_out::Gaussian{<:Any,<:BlocksOfDiagonalsPSD},
    x_pred::Gaussian{<:Any,<:BlocksOfDiagonalsPSD},
    obs::MatrixFreeLinearizedObservation,
    K::MatrixFreeGain,
    B;
    cache,
)
    H = obs.H
    c = H.cache
    Ξ = Random.randn!(integrator_rng(H.integ), c.Ξ)
    G = mul!(c.G, x_pred.Σ.R', Ξ)
    condition_samples!(G, H, K)
    estimate_block_diagonal_covariance!(x_out.Σ.R, G, (c.Y, c.M))
    return x_out
end

"""
    condition_samples!(G, H, K::MatrixFreeGain)

Condition the samples in the columns of `G` on the observation, by replacing each sample
``g`` with ``g - K H g``. If the samples have the covariance ``Σ``, they then have the
posterior covariance ``Σ^F = Σ - K H Σ``.
"""
function condition_samples!(G, H::MatrixFreeObservationOperator, K::MatrixFreeGain)
    (; HG, gtmp) = H.cache
    mul!(HG, H, G)
    W = krylov_solve!(K.S, HG, :covariance)
    for j in axes(G, 2)
        view(G, :, j) .-= mul!(gtmp, K.ΣHᵀ, view(W, :, j))
    end
    return G
end

"""
    estimate_block_diagonal_covariance!(R::BlocksOfDiagonals, G, (Y, M))

Estimate the block-diagonal part of a covariance from the samples in the `m` columns of `G`,
and write its square root into `R`. The estimate is the sample covariance
``\\frac{1}{m} G G^T`` without the covariances between the blocks. Block ``i`` uses the rows
`i:d:D` of `G`, which hold dimension ``i`` of every derivative. `Y` and `M` are buffers.
"""
function estimate_block_diagonal_covariance!(R::BlocksOfDiagonals, G, (Y, M))
    d, m = nblocks(R), size(G, 2)
    for (i, Ri) in enumerate(blocks(R))
        rmul!(transpose!(Y, view(G, i:d:size(G, 1), :)), 1 / sqrt(m))
        chol = cholesky!(Symmetric(_matmul!(M, Y', Y)), check=false)
        if issuccess(chol) && is_well_conditioned(chol.U)
            copy!(Ri, chol.U)
        else
            copy!(Ri, triangularize!(Y; cachemat=nothing))
        end
    end
    return R
end

# The error estimate needs the diagonal of H Q(h) Hᵀ with adaptive steps
function observed_process_noise!(
    cache::AbstractODEFilterCache, obs::MatrixFreeLinearizedObservation,
)
    H, adaptive = obs.H, obs.H.integ.opts.adaptive
    return MatrixFreeObservedCovariance(
        H,
        cache.Qh.R,
        H.cache.HQHᵀ;
        needs_diagonal=adaptive,
    )
end

invquad(z, S::MatrixFreeObservedCovariance; kwargs...) =
    dot(z, krylov_solve!(S, z, :calibration))

diag!(v::AbstractVector, S::MatrixFreeObservedCovariance) = copy!(v, S.diagonal)
