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
        (; ΣEt=(zeros(T, D), zeros(T, D)), preconditioner=dvec(), diagonal=dvec())
    return (;
        r=dvec(), u=dvec(),
        J_D=isnothing(linearization.diagonal_jacobian) ? nothing : dvec(),
        v=dvec(), Jv=dvec(), Jtw=dvec(), E0ΣHtw=dvec(), E0g=dvec(), E1g=dvec(), z=dvec(),
        HΣHt=observed_covariance(), HQHt=observed_covariance(),
        ξ=zeros(T, 2d), L=(dvec(), dvec(), dvec()),
        Ξ=zeros(T, D, m), G=zeros(T, D, m), HG=zeros(T, d, m),
        Gt_i=zeros(T, m, q + 1), Σ_i=zeros(T, q + 1, q + 1), KHg=zeros(T, D),
        cg=CgWorkspace(d, d, Vector{T}), W=zeros(T, d, m),
    )
end

"""
    AbstractMatrixFreeOperator

A minimal interface for matrix-free linear operators.

Operators are expected to implement a matrix-vector `mul!`:
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
    if !isnothing(c.J_D)
        linearization.diagonal_jacobian(c.J_D, c.u, p, t)
    end
    return LinearizedObservation(m, c.r, MatrixFreeObservationOperator(integ, c), R)
end

"""
    apply_jvp!(H::MatrixFreeObservationOperator, v)

For a given ``H = E_1 - J E_0`` with `H::MatrixFreeObservationOperator`,
compute ``J v`` as a Jacobian-vector product (JVP) by calling the `jvp` of
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
compute ``J^T w`` as a vector-Jacobian product (VJP) by calling the `vjp` of
the ODE function.

If `w` is not a `Vector`, it is first copied into one, because `f.vjp` may require it.
"""
function apply_vjp!(H::MatrixFreeObservationOperator, w)
    (; f, p, t, dt) = H.integ
    Jtw = H.cache.Jtw
    w = w isa Vector ? w : copy!(H.cache.v, w)
    f.vjp(Jtw, w, H.cache.u, p, t + dt)
    H.integ.sol.pnstats.nvjp += 1
    return Jtw
end

function mul!(y::AbstractVector, H::MatrixFreeObservationOperator, x::AbstractVector)
    d = length(y)
    E0x, E1x = view(x, 1:d), view(x, (d+1):2d)
    JE0x = apply_jvp!(H, E0x)
    @.. y = E1x - JE0x
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
    ΣE0t::V
    ΣE1t::V
    function MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, (ΣE0t, ΣE1t))
        d = nblocks(R)
        for (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
            ΣE0t[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 1))
            ΣE1t[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 2))
        end
        return new{typeof(H),typeof(ΣE0t)}(H, ΣE0t, ΣE1t)
    end
end

function mul!(x::AbstractVector, ΣHt::MatrixFreeCrossCovariance, w::AbstractVector)
    (; H, ΣE0t, ΣE1t) = ΣHt
    d = length(w)
    Jtw = apply_vjp!(H, w)
    for k in 1:(length(x)÷d)
        idx = ((k-1)*d+1):(k*d)
        @views x[idx] .= ΣE1t[idx] .* w .- ΣE0t[idx] .* Jtw
    end
    return x
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
if a user explicitly passed a function for the diagonal of the ODE Jacobian,
we use that to compute an approximated diagonal;
otherwise we estimate it stochastically.

As the local error estimate needs the stochastic diagonal, one can also compute both
by setting `needs_diagonal=true`.
"""
struct MatrixFreeObservedCovariance{CT,RT,V} <: AbstractMatrixFreeOperator
    ΣHt::CT
    R::RT
    preconditioner::V
    diagonal::V
    has_diagonal::Bool
    function MatrixFreeObservedCovariance(
        H, R::BlocksOfDiagonals, buffers; needs_diagonal::Bool,
    )
        ΣHt = MatrixFreeCrossCovariance(H, R, buffers.ΣEt)
        (; preconditioner, diagonal) = buffers
        J_D = H.cache.J_D
        has_diagonal = needs_diagonal || isnothing(J_D)
        S = new{typeof(ΣHt),typeof(R),typeof(diagonal)}(
            ΣHt, R, preconditioner, diagonal, has_diagonal,
        )
        if has_diagonal
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
    (; H, ΣE0t, ΣE1t) = S.ΣHt
    d = length(w)
    Jtw = apply_vjp!(H, w)
    E0ΣHtw = H.cache.E0ΣHtw
    @views E0ΣHtw .= ΣE1t[1:d] .* w .- ΣE0t[1:d] .* Jtw
    @views y .= ΣE1t[(d+1):2d] .* w .- ΣE0t[(d+1):2d] .* Jtw
    y .-= apply_jvp!(H, E0ΣHtw)
    return y
end

"""
    estimate_diagonal!(out, S::MatrixFreeObservedCovariance, n)

Estimate the diagonal of ``S = H Σ H^T`` stochastically, from `n` probes.

For samples ``g`` with the covariance ``Σ``, the expectation of the elementwise square ``(H g)^2`` is the diagonal of ``S``.
We therefore average ``(H g)^2`` over `n` samples, which is an estimator of the Hutchinson type.
Each sample costs one Jacobian-vector product.

Since ``H`` only reads ``E_0 g`` and ``E_1 g``, we only sample these.
And since ``Σ`` is block-diagonal, the pairs ``(u_i, u_i')`` of different dimensions are independent, so we draw each pair from its own ``2 × 2`` covariance in ``Σ``.
"""
function estimate_diagonal!(out, S::MatrixFreeObservedCovariance, n)
    (; H, ΣE0t, ΣE1t) = S.ΣHt
    c = H.cache
    d = length(out)
    # The Cholesky factor [l00 0; l10 l11] of the covariance of (uᵢ, uᵢ')
    Σ00, Σ01, Σ11 = view(ΣE0t, 1:d), view(ΣE1t, 1:d), view(ΣE1t, (d+1):2d)
    l00, l10, l11 = c.L
    @.. l00 = sqrt(Σ00)
    @.. l10 = ifelse(l00 > 0, Σ01 / l00, zero(l00))
    @.. l11 = sqrt(max(Σ11 - l10^2, zero(l10)))
    ξ0, ξ1 = view(c.ξ, 1:d), view(c.ξ, (d+1):2d)
    (; E0g, E1g) = c
    rng = integrator_rng(H.integ)
    fill!(out, 0)
    for _ in 1:n
        Random.randn!(rng, c.ξ)
        @.. E0g = l00 * ξ0
        @.. E1g = l10 * ξ0 + l11 * ξ1
        JE0g = apply_jvp!(H, E0g)
        @.. out = out + (E1g - JE0g)^2
    end
    @.. out = out / n
    return out
end

"""
    estimate_diagonal_with_diagonal_jacobian!(out, S::MatrixFreeObservedCovariance, J_D)

Estimate the diagonal of ``S = H Σ H^T`` deterministically, by the diagonal of ``H_D Σ H_D^T`` with ``H_D = E_1 - J_D E_0``, where ``J_D`` is the diagonal of the Jacobian.

This needs no Jacobian-vector products, but it ignores the off-diagonal entries of ``J``.
Since row ``i`` of ``H_D`` only reads ``u_i`` and ``u_i'``, its entry only needs the ``i``-th block of the square root ``R`` of ``Σ``.
"""
function estimate_diagonal_with_diagonal_jacobian!(
    out, S::MatrixFreeObservedCovariance, J_D,
)
    for (i, Ri) in enumerate(blocks(S.R))
        # (H_D Σ H_Dᵀ)ᵢᵢ = ‖R H_Dᵀ eᵢ‖² = ‖r₁ - J_Dᵢ r₀‖², with the columns r₀ and r₁ of Rᵢ
        out[i] = sum(k -> (Ri[k, 2] - J_D[i] * Ri[k, 1])^2, axes(Ri, 1))
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

Solve ``S x = b`` by CG with the Jacobi preconditioner of `S`, and return ``x``.

If `b` is a matrix, we solve for each column separately.
`part` is one of `:mean`, `:covariance` and `:calibration`.
It sets the tolerance, and the solve counts in its statistics.

If `b` or the preconditioner is not finite, or if CG does not converge, we throw a `LinearSolveFailure`, which rejects the step.
"""
function krylov_solve!(
    S::MatrixFreeObservedCovariance, b::AbstractVector, part; itmax=MAX_CG_ITERATIONS,
)
    throw_if_not_finite(S, b)
    H = S.ΣHt.H
    workspace = H.cache.cg
    rtol = krylov_rtol(H, part)
    M = Diagonal(S.preconditioner)
    cg!(workspace, S, b; M, ldiv=true, atol=zero(rtol), rtol, itmax)
    count_solve!(H, part, Krylov.iteration_count(workspace))
    Krylov.issolved(workspace) || throw(LinearSolveFailure())
    return Krylov.solution(workspace)
end
function krylov_solve!(S::MatrixFreeObservedCovariance, B::AbstractMatrix, part)
    W = S.ΣHt.H.cache.W
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

The Kalman gain ``K = Σ H^T S^{-1}``, stored as its two factors ``Σ H^T`` and ``S``.

`update_mean!` returns it, so that `update_cov!` can reuse ``S`` and its preconditioner instead of building them again.
"""
struct MatrixFreeGain{CT,ST}
    ΣHt::CT
    S::ST
end
MatrixFreeGain(S::MatrixFreeObservedCovariance) = MatrixFreeGain(S.ΣHt, S)

"""
    update_mean!(x_out, x_pred, obs::MatrixFreeLinearizedObservation; cache)

Update the mean on a matrix-free observation.

We solve ``S w = z`` for ``S = H Σ H^T`` by CG, where ``z`` is the residual at the predicted mean ``μ``, and set ``μ^F = μ - Σ H^T w``.
The log-likelihood would need the determinant of ``S``, which we do not compute, so we return it as `NaN`.
We also return the gain ``K`` for `update_cov!`.
"""
function update_mean!(x_out, x_pred, obs::MatrixFreeLinearizedObservation; cache)
    (; m, H) = obs
    c = H.cache
    S = MatrixFreeObservedCovariance(H, x_pred.Σ.R, c.HΣHt; needs_diagonal=false)
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
    mul!(x_out.μ, S.ΣHt, w)
    @.. x_out.μ = x_pred.μ - x_out.μ
    K = MatrixFreeGain(S)
    return (; loglikelihood=oftype(ztSinvz, NaN), ztSinvz, S, z, K, B=nothing)
end

"""
    update_cov!(x_out, x_pred, obs::MatrixFreeLinearizedObservation, K, B; cache)

Update the covariance on a matrix-free observation, approximately, from samples.

The exact posterior covariance ``Σ^F = Σ - K H Σ`` is not block-diagonal, because ``J`` couples the dimensions.
We approximate it by its block-diagonal part, which we estimate from `nprobes` samples.
We draw samples ``g`` with the covariance ``Σ`` and condition them on the observation with `condition_samples!`, so that they have the covariance ``Σ^F``.
Then `estimate_block_diagonal_covariance!` computes the block-diagonal part of their sample covariance.
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
    Ξ = Random.randn!(integrator_rng(H.integ), c.Ξ)  # ~ N(0, I)
    G = mul!(c.G, x_pred.Σ.R', Ξ)                    # ~ N(0, Σ)
    condition_samples!(G, H, K)                      # ~ N(0, Σ - K H Σ)
    estimate_block_diagonal_covariance!(x_out.Σ.R, G, (c.Gt_i, c.Σ_i))
    return x_out
end

"""
    condition_samples!(G, H, K::MatrixFreeGain)

Condition the samples in the columns of `G` on the observation, by replacing each sample ``g`` with ``g - K H g``.

If the samples have the covariance ``Σ``, the results have the posterior covariance ``Σ^F = Σ - K H Σ``.
Each sample costs one linear solve with ``S``.
"""
function condition_samples!(G, H::MatrixFreeObservationOperator, K::MatrixFreeGain)
    (; HG, KHg) = H.cache
    mul!(HG, H, G)
    W = krylov_solve!(K.S, HG, :covariance)
    for j in axes(G, 2)
        view(G, :, j) .-= mul!(KHg, K.ΣHt, view(W, :, j))
    end
    return G
end

"""
    estimate_block_diagonal_covariance!(R::BlocksOfDiagonals, G, (Gt_i, Σ_i))

Estimate the block-diagonal part of a covariance from the samples in the columns of `G`, and write its square root into `R`.

The samples have mean zero, so their sample covariance is ``\\frac{1}{m} G G^T`` for `m` samples.
We keep only its blocks.
Block ``i`` is the covariance of dimension ``i`` across all derivatives, so it uses the rows `i:d:D` of `G`.
We compute its square root by a Cholesky factorization, or by a QR decomposition if the Cholesky factorization fails or is ill-conditioned.
`Gt_i` and `Σ_i` are buffers.
"""
function estimate_block_diagonal_covariance!(R::BlocksOfDiagonals, G, (Gt_i, Σ_i))
    d, m = nblocks(R), size(G, 2)
    for (i, Ri) in enumerate(blocks(R))
        rmul!(transpose!(Gt_i, view(G, i:d:size(G, 1), :)), 1 / sqrt(m))
        chol = cholesky!(Symmetric(_matmul!(Σ_i, Gt_i', Gt_i)), check=false)
        if issuccess(chol) && is_well_conditioned(chol.U)
            copy!(Ri, chol.U)
        else
            copy!(Ri, triangularize!(Gt_i; cachemat=nothing))
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
        H.cache.HQHt;
        needs_diagonal=adaptive,
    )
end

invquad(z, S::MatrixFreeObservedCovariance; kwargs...) =
    dot(z, krylov_solve!(S, z, :calibration))

function diag!(v::AbstractVector, S::MatrixFreeObservedCovariance)
    @assert S.has_diagonal "Build `S` with `needs_diagonal=true` to estimate its diagonal."
    return copy!(v, S.diagonal)
end
