# The `MatfreeEK1`. It accesses the Jacobian of the ODE only through `f.jvp` and `f.vjp`,
# solves its linear systems with Krylov methods, and estimates a block-diagonal posterior
# covariance from samples.

# Probes for the diagonals of `H Σ Hᵀ` and `H Q(h) Hᵀ`. Each costs one JVP.
const DIAGONAL_PROBES = 32
const CALIBRATION_RTOL = 0.1
const MAX_CG_ITERATIONS = 10_000
const MAX_BLOCK_GMRES_ITERATIONS = 200

"""
    throw_if_unsupported(::MatrixFreeJacobian, alg, f)

Throw an `ArgumentError` if the `MatfreeEK1` does not support the settings of `alg` or the
ODE function `f`. It does not support second-order ODEs, mass matrices, observation noise or
multivariate diffusion models. It needs `jvp` and `vjp` in the ODE function, and at least as
many probes as there are derivatives in the state.
"""
function throw_if_unsupported(linearization::MatrixFreeJacobian, alg, f)
    unsupported(what) = ArgumentError("The `MatfreeEK1` does not support $what.")
    if f isa DynamicalODEFunction
        throw(unsupported("second-order ODEs"))
    elseif f.mass_matrix !== I
        throw(unsupported("mass matrices"))
    elseif !isnothing(alg.pn_observation_noise)
        throw(unsupported("observation noise"))
    elseif !(alg.diffusionmodel isa Union{DynamicDiffusion,FixedDiffusion})
        throw(unsupported("multivariate diffusion models"))
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
    m = number_of_probes(linearization, q)
    if m < q + 1
        throw(
            ArgumentError(
                "`nprobes` must be at least the number of derivatives in the state, " *
                "$(q + 1), but it is $m."),
        )
    end
    return nothing
end

# The number of probes for the posterior covariance. By default, it is twice the number of
# derivatives in the state.
number_of_probes(linearization::MatrixFreeJacobian, q) =
    isnothing(linearization.nprobes) ? 2(q + 1) : linearization.nprobes

function make_linearization_cache(
    linearization::MatrixFreeJacobian, C::BlockDiagonalCovariance{T}, u, f,
) where {T}
    d, q = C.d, C.q
    D = d * (q + 1)
    m = number_of_probes(linearization, q)
    dvec() = zeros(T, d)
    observed_covariance() =
        (; ΣEᵀ=(zeros(T, D), zeros(T, D)), preconditioner=dvec(), diagonal=dvec())
    # Block-GMRES needs at least twice as many rows as right-hand sides
    has_block_gmres = 2m <= d
    return (;
        r=dvec(), u=dvec(),
        diagonal_jacobian=isnothing(linearization.diagonal_jacobian) ? nothing : dvec(),
        v=dvec(), Jv=dvec(), a=dvec(), e0=dvec(), z=dvec(),
        HΣHᵀ=observed_covariance(), HQHᵀ=observed_covariance(), sqrtP=dvec(),
        ξ=zeros(T, 2d), L=(dvec(), dvec(), dvec()),
        Ξ=zeros(T, D, m), G=zeros(T, D, m), HG=zeros(T, d, m),
        Y=zeros(T, m, q + 1), M=zeros(T, q + 1, q + 1), gtmp=zeros(T, D),
        cg=CgWorkspace(d, d, Vector{T}),
        block_gmres=has_block_gmres ?
                    BlockGmresWorkspace(d, d, m, Vector{T}, Matrix{T}; memory=20) : nothing,
        W=has_block_gmres ? nothing : zeros(T, d, m),
    )
end

"""
    AbstractMatrixFreeOperator

A linear operator that is applied with `mul!`, without storing it as a matrix. Applied to a
matrix, it acts on each column.
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

The observation matrix ``H = E_1 - J E_0`` of the ODE, as an operator. ``E_0`` and ``E_1``
select the solution and its first derivative from the state. ``J`` is the Jacobian of the
ODE at the solution `cache.u`, and it is never formed. So ``H x = E_1 x - J E_0 x`` costs
one Jacobian-vector product, from `jvp!`. The cross-covariance ``Σ H^T`` needs products
with ``J^T`` instead, from `vjp!`.

The integrator `integ` provides the ODE function, its parameters and the time. It also
provides the random number generator for the probes, the tolerance of the linear solves,
and the statistics that count the products and the solves. `cache` holds the buffers of
the linearization.
"""
struct MatrixFreeObservationOperator{I,C} <: AbstractMatrixFreeOperator
    integ::I
    cache::C
end

const MatrixFreeLinearizedObservation =
    LinearizedObservation{<:Any,<:Any,<:MatrixFreeObservationOperator}

# The `rng` that `solve` takes on OrdinaryDiffEq 7. Older versions have none.
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
    jvp!(Jv, H::MatrixFreeObservationOperator, v)

Multiply `v` with the Jacobian ``J`` of the ODE at the point of `H`, and write the result
into `Jv`. This calls the `jvp` of the ODE function. The products with ``H`` and with
``H Σ H^T`` use it, and so does the estimate of the diagonal of ``H Σ H^T`` from probes.
If `v` is not a `Vector`, it is first copied into one, because `f.jvp` may require it.
"""
function jvp!(Jv, H::MatrixFreeObservationOperator, v)
    (; f, p, t, dt) = H.integ
    v = v isa Vector ? v : copy!(H.cache.v, v)
    f.jvp(Jv, v, H.cache.u, p, t + dt)
    H.integ.sol.pnstats.njvp += 1
    return Jv
end

"""
    vjp!(Jᵀw, H::MatrixFreeObservationOperator, w)

Multiply `w` with the transpose ``J^T`` of the Jacobian of the ODE at the point of `H`, and
write the result into `Jᵀw`. This calls the `vjp` of the ODE function. The products with
``Σ H^T`` and with ``H Σ H^T`` use it. If `w` is not a `Vector`, it is first copied into
one, because `f.vjp` may require it.
"""
function vjp!(Jᵀw, H::MatrixFreeObservationOperator, w)
    (; f, p, t, dt) = H.integ
    w = w isa Vector ? w : copy!(H.cache.v, w)
    f.vjp(Jᵀw, w, H.cache.u, p, t + dt)
    H.integ.sol.pnstats.nvjp += 1
    return Jᵀw
end

# H x = E₁ x - J E₀ x
function mul!(y::AbstractVector, H::MatrixFreeObservationOperator, x::AbstractVector)
    d = length(y)
    E₀x, E₁x = view(x, 1:d), view(x, (d+1):2d)
    JE₀x = jvp!(H.cache.Jv, H, E₀x)
    @.. y = E₁x - JE₀x
    return y
end

"""
    MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, buffers)

The cross-covariance ``Σ H^T`` between the state and the observation, as an operator, for
the block-diagonal covariance ``Σ = R^T R``. Since ``H = E_1 - J E_0``, it is
``Σ H^T w = Σ E_1^T w - Σ E_0^T J^T w``. The constructor computes ``Σ E_0^T`` and
``Σ E_1^T`` into `buffers`. These are the columns of each block of ``Σ`` for the solution
and its first derivative. Each has one nonzero entry per row, so it is stored as a vector
of the length of the state.
"""
struct MatrixFreeCrossCovariance{HT,V} <: AbstractMatrixFreeOperator
    H::HT
    ΣE₀ᵀ::V
    ΣE₁ᵀ::V
    function MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, (ΣE₀ᵀ, ΣE₁ᵀ))
        d = nblocks(R)
        for (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
            # The entries (k, 1) and (k, 2) of the i-th block Rᵢᵀ Rᵢ of Σ
            ΣE₀ᵀ[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 1))
            ΣE₁ᵀ[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 2))
        end
        return new{typeof(H),typeof(ΣE₀ᵀ)}(H, ΣE₀ᵀ, ΣE₁ᵀ)
    end
end

# Σ Hᵀ w = Σ E₁ᵀ w - Σ E₀ᵀ Jᵀ w, for one derivative of the state after the other
function mul!(out::AbstractVector, C::MatrixFreeCrossCovariance, w::AbstractVector)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = C
    d = length(w)
    Jᵀw = vjp!(H.cache.a, H, w)
    for k in 0:(length(out)÷d-1)
        idx = (k*d+1):((k+1)*d)
        @views out[idx] .= ΣE₁ᵀ[idx] .* w .- ΣE₀ᵀ[idx] .* Jᵀw
    end
    return out
end

"""
    MatrixFreeObservedCovariance(H, R::BlocksOfDiagonals, buffers; needs_diagonal)

The covariance ``H Σ H^T`` of the observation, as an operator for Krylov.jl, for the
block-diagonal covariance ``Σ = R^T R``. The update solves with it for the predicted
covariance ``Σ``, and the calibration for ``Σ = Q(h)``.

The linear solves need a Jacobi preconditioner, so the constructor computes one, in
`preconditioner`. It is an approximation of the diagonal. With a diagonal Jacobian, it is
the diagonal for ``J`` replaced by its diagonal, which costs no product. Without, it is
estimated from probes. The error estimate needs the diagonal from probes in any case. With
`needs_diagonal`, the constructor estimates it also then, into `diagonal`, which `diag!`
returns.
"""
struct MatrixFreeObservedCovariance{CT,RT,V} <: AbstractMatrixFreeOperator
    C::CT
    R::RT
    preconditioner::V
    diagonal::V
    function MatrixFreeObservedCovariance(
        H, R::BlocksOfDiagonals, buffers; needs_diagonal::Bool,
    )
        C = MatrixFreeCrossCovariance(H, R, buffers.ΣEᵀ)
        (; preconditioner, diagonal) = buffers
        S = new{typeof(C),typeof(R),typeof(diagonal)}(C, R, preconditioner, diagonal)
        J_D = H.cache.diagonal_jacobian
        if needs_diagonal || isnothing(J_D)
            estimate_diagonal!(diagonal, S, DIAGONAL_PROBES)
        end
        if isnothing(J_D)
            copy!(preconditioner, diagonal)
        else
            diagonal_with_diagonal_jacobian!(preconditioner, S, J_D)
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
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = S.C
    d = length(w)
    Jᵀw = vjp!(H.cache.a, H, w)
    E₀ΣHᵀw = H.cache.e0
    @views E₀ΣHᵀw .= ΣE₁ᵀ[1:d] .* w .- ΣE₀ᵀ[1:d] .* Jᵀw
    @views y .= ΣE₁ᵀ[(d+1):2d] .* w .- ΣE₀ᵀ[(d+1):2d] .* Jᵀw
    y .-= jvp!(H.cache.Jv, H, E₀ΣHᵀw)
    return y
end

"""
    draw_probes!(ξ, rng)

Fill `ξ` with probes, which are independent standard normal entries. The estimators from
probes only need ``E[ξ ξ^T] = I``, so Rademacher entries would work as well.
"""
draw_probes!(ξ, rng) = Random.randn!(rng, ξ)

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
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = S.C
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
        draw_probes!(c.ξ, rng)
        @.. E₀g = l₀₀ * ξ₀
        @.. E₁g = l₁₀ * ξ₀ + l₁₁ * ξ₁
        # (H g)² = (E₁ g - J E₀ g)²
        JE₀g = jvp!(c.Jv, H, E₀g)
        @.. out = out + (E₁g - JE₀g)^2
    end
    @.. out = out / n
    return out
end

"""
    diagonal_with_diagonal_jacobian!(out, S::MatrixFreeObservedCovariance, J_D)

The diagonal of ``H_D Σ H_D^T``, with ``H_D = E_1 - J_D E_0`` for the diagonal ``J_D`` of
the Jacobian. Row ``i`` of ``H_D`` only reads ``u_i`` and ``u_i'``, so its entry only needs
the ``i``-th block of the square root ``R`` of ``Σ``.
"""
function diagonal_with_diagonal_jacobian!(out, S::MatrixFreeObservedCovariance, J_D)
    for (i, Ri) in enumerate(blocks(S.R))
        # (H_D Σ H_Dᵀ)ᵢᵢ = ‖R H_Dᵀ eᵢ‖² = ‖r₁ - J_Dᵢ r₀‖², with the columns r₀ and r₁ of Rᵢ
        out[i] = sum(l -> (Ri[l, 2] - J_D[i] * Ri[l, 1])^2, axes(Ri, 1))
    end
    return out
end

# The relative tolerance of the linear solves. The calibration needs less accuracy than the
# update, whose tolerance is the solver's `reltol`, but at most 1e-2. It is at least
# `sqrt(eps)`, because the Krylov methods cannot reach a residual of zero, as a `reltol` of
# zero would ask for.
function krylov_rtol(H::MatrixFreeObservationOperator, part)
    T = eltype(H.cache.u)
    part === :calibration && return T(CALIBRATION_RTOL)
    return clamp(T(minimum(H.integ.opts.reltol)), sqrt(eps(T)), T(1e-2))
end

struct LinearSolveFailure <: Exception end

"""
    krylov_solve!(S::MatrixFreeObservedCovariance, b, part)

Solve ``S x = b`` with a Krylov method, and return ``x``. `part` is `:mean`, `:covariance`
or `:calibration`. It sets the tolerance, and the solve counts in its statistics. The
Jacobi preconditioner is the one of `S`. For a vector `b`, this uses CG. For a matrix `b`,
it uses block-GMRES on all columns at once, or CG on each column if there are more than
half as many columns as rows. If `b` or the preconditioner is not finite, or if the solve
does not converge, it throws a `LinearSolveFailure`.
"""
function krylov_solve!(
    S::MatrixFreeObservedCovariance, b::AbstractVector, part; itmax=MAX_CG_ITERATIONS,
)
    throw_if_not_finite(S, b)
    H = S.C.H
    workspace = H.cache.cg
    rtol = krylov_rtol(H, part)
    M = Diagonal(S.preconditioner)
    cg!(workspace, S, b; M, ldiv=true, atol=zero(rtol), rtol, itmax)
    count_solve!(H, part, Krylov.iteration_count(workspace))
    Krylov.issolved(workspace) || throw(LinearSolveFailure())
    return Krylov.solution(workspace)
end
function krylov_solve!(S::MatrixFreeObservedCovariance, B::AbstractMatrix, part)
    H = S.C.H
    c = H.cache
    if isnothing(c.block_gmres)
        for j in axes(B, 2)
            copy!(view(c.W, :, j), krylov_solve!(S, view(B, :, j), part))
        end
        return c.W
    end
    throw_if_not_finite(S, B)
    rtol = krylov_rtol(H, part)
    # Preconditioned on both sides, so that the residual is weighted as in the
    # preconditioned CG
    @.. c.sqrtP = sqrt(S.preconditioner)
    M = Diagonal(c.sqrtP)
    block_gmres!(c.block_gmres, S, B; M, N=M, ldiv=true, atol=zero(rtol), rtol,
        itmax=MAX_BLOCK_GMRES_ITERATIONS)
    count_solve!(H, part, Krylov.iteration_count(c.block_gmres))
    Krylov.issolved(c.block_gmres) || throw(LinearSolveFailure())
    return Krylov.solution(c.block_gmres)
end

# Krylov.jl throws its own errors for NaN, which would end the solve instead of rejecting the
# step, and an infinite preconditioner makes it skip the solve
function throw_if_not_finite(S::MatrixFreeObservedCovariance, b)
    if !(all(isfinite, b) && all(isfinite, S.preconditioner))
        throw(LinearSolveFailure())
    end
    return nothing
end

# Count a linear solve in the statistics of `part`
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
    MatrixFreeGain(S)

The Kalman gain ``K = Σ H^T S^{-1}``. `S` holds the cross-covariance ``Σ H^T`` and the
Jacobi preconditioner for the solves with ``S``. `update_mean!` returns the gain for
`update_cov!`.
"""
struct MatrixFreeGain{ST}
    S::ST
end

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
    mul!(x_out.μ, S.C, w)
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
    Ξ = draw_probes!(c.Ξ, integrator_rng(H.integ))
    G = mul!(c.G, x_pred.Σ.R', Ξ)
    condition_samples!(G, H, K)
    estimate_block_diagonal_covariance!(x_out.Σ.R, G, (c.Y, c.M))
    return x_out
end

"""
    condition_samples!(G, H, K::MatrixFreeGain)

Condition the samples in the columns of `G` on the observation, by replacing each sample
``g`` with ``g - K H g``. If the samples have the covariance ``Σ``, they then have the
posterior covariance ``Σ^F = Σ - K H Σ``. The linear systems of all samples are solved at
once.
"""
function condition_samples!(G, H::MatrixFreeObservationOperator, K::MatrixFreeGain)
    (; HG, gtmp) = H.cache
    mul!(HG, H, G)
    W = krylov_solve!(K.S, HG, :covariance)
    for j in axes(G, 2)
        view(G, :, j) .-= mul!(gtmp, K.S.C, view(W, :, j))
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

# zᵀ (H Q(h) Hᵀ)⁻¹ z, for the calibration
invquad(z, S::MatrixFreeObservedCovariance; kwargs...) =
    dot(z, krylov_solve!(S, z, :calibration))

diag!(v::AbstractVector, S::MatrixFreeObservedCovariance) = copy!(v, S.diagonal)
