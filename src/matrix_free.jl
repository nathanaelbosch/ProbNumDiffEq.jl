# The linearization of the `MatfreeEK1` and the update on its observation: the Jacobian of
# the ODE is only accessed through `f.jvp` and `f.vjp`, the linear systems are solved with
# Krylov methods, and the posterior covariance is block-diagonal, estimated from probes.

# Probes for the diagonals of `H Σ Hᵀ` and `H Q(h) Hᵀ`; they cost one JVP each, no solve
const DIAGONAL_PROBES = 32
const CALIBRATION_RTOL = 0.1
const MAX_CG_ITERATIONS = 10_000
const MAX_BLOCK_GMRES_ITERATIONS = 200

"""
    throw_if_unsupported(::MatrixFreeJacobian, alg, f)

Throw an `ArgumentError` for what the `MatfreeEK1` does not support: second-order ODEs,
mass matrices, observation noise, multivariate diffusion models, an ODE function without
`jvp` or `vjp`, and fewer probes than derivatives in the state.
"""
function throw_if_unsupported(linearization::MatrixFreeJacobian, alg, f)
    (; diffusionmodel, pn_observation_noise, prior) = alg
    unsupported = String[]
    f isa DynamicalODEFunction && push!(unsupported, "second-order ODEs")
    f.mass_matrix === I || push!(unsupported, "mass matrices")
    isnothing(pn_observation_noise) || push!(unsupported, "`pn_observation_noise`")
    diffusionmodel isa Union{DynamicDiffusion,FixedDiffusion} ||
        push!(unsupported, "`diffusionmodel = $(repr(diffusionmodel))`")
    isempty(unsupported) || throw(
        ArgumentError(
            "The `MatfreeEK1` does not support $(join(unsupported, " or ")) yet."),
    )
    missing_products = [string("`", name, "`") for name in (:jvp, :vjp)
                                if isnothing(getproperty(f, name))]
    isempty(missing_products) || throw(
        ArgumentError(
            "The `MatfreeEK1` needs the Jacobian-vector product `jvp(Jv, v, u, p, t)` " *
            "and the vector-Jacobian product `vjp(Jᵀv, v, u, p, t)` of the ODE function, " *
            "but $(join(missing_products, " and ")) is missing. Pass them as " *
            "`ODEFunction(f; jvp, vjp)`."),
    )
    q = num_derivatives(prior)
    m = something(linearization.nprobes, 2(q + 1))
    m >= q + 1 || throw(
        ArgumentError(
            "`nprobes` must be at least the number of derivatives in the " *
            "state, $(q + 1), but it is $m.",
        ),
    )
    return nothing
end

function make_linearization_cache(
    linearization::MatrixFreeJacobian, C::CovarianceStructure{T}, u, f,
) where {T}
    @assert C isa BlockDiagonalCovariance
    d, q = C.d, C.q
    D = d * (q + 1)
    m = something(linearization.nprobes, 2(q + 1))
    vec() = zeros(T, d)
    observed_covariance() =
        (; ΣEᵀ=(zeros(T, D), zeros(T, D)), diagonal=vec())
    has_block_gmres = 2m <= d
    return (;
        r=vec(), u=vec(),
        diagonal_jacobian=isnothing(linearization.diagonal_jacobian) ? nothing : vec(),
        v=vec(), Jv=vec(), a=vec(), e0=vec(), z=vec(),
        HΣHᵀ=observed_covariance(), HQHᵀ=observed_covariance(),
        P=vec(), sqrtP=vec(), PQ=vec(),
        ξ=zeros(T, D), Ξ=zeros(T, D, m), G=zeros(T, D, m), HG=zeros(T, d, m),
        Y=zeros(T, m, q + 1), M=zeros(T, q + 1, q + 1), gtmp=zeros(T, D),
        cg=CgWorkspace(d, d, Vector{T}),
        block_gmres=has_block_gmres ?
                    BlockGmresWorkspace(d, d, m, Vector{T}, Matrix{T}; memory=20) : nothing,
        W=has_block_gmres ? nothing : zeros(T, d, m),
    )
end

"""
    AbstractMatrixFreeOperator

An operator of the `MatfreeEK1` that is only applied, through `mul!`: the observation
matrix ``H``, the cross-covariance ``Σ H^T``, or ``H Σ H^T``. A product with a matrix
applies it column by column.
"""
abstract type AbstractMatrixFreeOperator end

function mul!(Y::AbstractMatrix, A::AbstractMatrixFreeOperator, X::AbstractMatrix)
    for j in axes(X, 2)
        mul!(view(Y, :, j), A, view(X, :, j))
    end
    return Y
end

"""
    MatrixFreeObservationOperator(u, integ, cache)

The observation matrix ``H = E_1 - J E_0`` as an operator, with the Jacobian ``J`` of the
ODE at `u`: products with ``J`` and ``J^T`` call `f.jvp` and `f.vjp` of the integrator
`integ`. Its random number generator draws the probes, its `reltol` sets the tolerance of
the linear solves, and its statistics count the products and the solves. `cache` holds the
buffers of the linearization.
"""
struct MatrixFreeObservationOperator{U,I,C} <: AbstractMatrixFreeOperator
    u::U
    integ::I
    cache::C
end

const MatrixFreeLinearizedObservation =
    LinearizedObservation{<:Any,<:Any,<:MatrixFreeObservationOperator}

# The `rng` that `solve` takes on OrdinaryDiffEq 7; older versions have none
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
    isnothing(c.diagonal_jacobian) ||
        linearization.diagonal_jacobian(c.diagonal_jacobian, c.u, p, t)
    return LinearizedObservation(m, c.r, MatrixFreeObservationOperator(c.u, integ, c), R)
end

"""
    jvp!(Jv, H::MatrixFreeObservationOperator, v)

``J v`` for the Jacobian ``J`` of the ODE in ``H = E_1 - J E_0``, from `f.jvp`. The products
with ``H`` and with ``H Σ H^T`` use it, and so does the probe estimate of the diagonal of
``H Σ H^T``. `v` is copied into a buffer first, since `f.jvp` may require a `Vector`.
"""
function jvp!(out, H::MatrixFreeObservationOperator, v)
    (; f, p, t, dt) = H.integ
    f.jvp(out, copy!(H.cache.v, v), H.u, p, t + dt)
    H.integ.sol.pnstats.njvp += 1
    return out
end

"""
    vjp!(Jᵀw, H::MatrixFreeObservationOperator, w)

``J^T w`` for the Jacobian ``J`` of the ODE in ``H = E_1 - J E_0``, from `f.vjp`. The
products with ``Σ H^T`` and with ``H Σ H^T`` use it. `w` is copied into a buffer first,
since `f.vjp` may require a `Vector`.
"""
function vjp!(out, H::MatrixFreeObservationOperator, w)
    (; f, p, t, dt) = H.integ
    f.vjp(out, copy!(H.cache.v, w), H.u, p, t + dt)
    H.integ.sol.pnstats.nvjp += 1
    return out
end

# H x = E₁ x - J E₀ x
function mul!(y::AbstractVector, H::MatrixFreeObservationOperator, x::AbstractVector)
    d = length(y)
    @views jvp!(y, H, x[1:d])
    @views y .= x[(d+1):2d] .- y
    return y
end

"""
    MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, buffers)

``Σ H^T`` as an operator, for the block-diagonal ``Σ = R^T R``:
``Σ H^T w = Σ E_1^T w - Σ E_0^T J^T w``. ``Σ E_0^T`` and ``Σ E_1^T`` are the columns of each
block of ``Σ`` for the derivatives 0 and 1; they have one nonzero per row, so they are
stored as vectors, in `buffers`.
"""
struct MatrixFreeCrossCovariance{HT,V} <: AbstractMatrixFreeOperator
    H::HT
    ΣE₀ᵀ::V
    ΣE₁ᵀ::V
end
function MatrixFreeCrossCovariance(H, R::BlocksOfDiagonals, (ΣE₀ᵀ, ΣE₁ᵀ))
    d = nblocks(R)
    for (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
        ΣE₀ᵀ[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 1))
        ΣE₁ᵀ[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 2))
    end
    return MatrixFreeCrossCovariance(H, ΣE₀ᵀ, ΣE₁ᵀ)
end

# Σ Hᵀ w = Σ E₁ᵀ w - Σ E₀ᵀ (Jᵀ w)
function mul!(out::AbstractVector, C::MatrixFreeCrossCovariance, w::AbstractVector)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = C
    a = H.cache.a
    d = length(a)
    vjp!(a, H, w)
    @.. a = -a
    for k in 0:(length(out)÷d-1)
        idx = (k*d+1):((k+1)*d)
        @views out[idx] .= ΣE₀ᵀ[idx] .* a .+ ΣE₁ᵀ[idx] .* w
    end
    return out
end

"""
    MatrixFreeObservedCovariance(H, R::BlocksOfDiagonals, buffers; estimate_diagonal)

``H Σ H^T`` as an operator for Krylov.jl, for the block-diagonal ``Σ = R^T R``: the
covariance ``Σ`` seen through ``H``, as ``S`` in the update, or ``H Q(h) H^T`` from
[`observed_process_noise!`](@ref). Its field `C` is the cross-covariance ``Σ H^T``. With
`estimate_diagonal`, its diagonal is estimated from probes when it is built, into the field
`diagonal`, which `diag!` and the Jacobi preconditioner read; without, `diagonal` holds no
valid values.
"""
struct MatrixFreeObservedCovariance{CT,V} <: AbstractMatrixFreeOperator
    C::CT
    diagonal::V
end
function MatrixFreeObservedCovariance(
    H, R::BlocksOfDiagonals, buffers; estimate_diagonal::Bool,
)
    C = MatrixFreeCrossCovariance(H, R, buffers.ΣEᵀ)
    estimate_diagonal && estimate_diagonal!(buffers.diagonal, H, R, DIAGONAL_PROBES)
    return MatrixFreeObservedCovariance(C, buffers.diagonal)
end
size(S::MatrixFreeObservedCovariance) = (length(S.C.H.u), length(S.C.H.u))
size(S::MatrixFreeObservedCovariance, i::Integer) = length(S.C.H.u)
eltype(S::MatrixFreeObservedCovariance) = eltype(S.C.H.u)

# H Σ Hᵀ w = E₁ Σ Hᵀ w - J E₀ Σ Hᵀ w, forming only the derivatives 0 and 1 of Σ Hᵀ w
function mul!(y::AbstractVector, S::MatrixFreeObservedCovariance, w::AbstractVector)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = S.C
    (; a, e0, Jv) = H.cache
    d = length(a)
    vjp!(a, H, w)
    @.. a = -a
    @views e0 .= ΣE₀ᵀ[1:d] .* a .+ ΣE₁ᵀ[1:d] .* w
    @views y .= ΣE₀ᵀ[(d+1):2d] .* a .+ ΣE₁ᵀ[(d+1):2d] .* w
    jvp!(Jv, H, e0)
    y .-= Jv
    return y
end

"""
    draw_standard_probes!(ξ, rng)

Fill `ξ` with independent standard normal entries. The estimators from probes only need
``E[ξ ξ^T] = I``, so other distributions, such as Rademacher, would do as well.
"""
draw_standard_probes!(ξ, rng) = Random.randn!(rng, ξ)

"""
    sample_probes!(G, Ξ, R::BlocksOfDiagonals, rng)

The columns ``g = R^T ξ`` of `G`, which have the covariance ``R^T R``, from standard probes
``ξ`` drawn into the columns of `Ξ`.
"""
function sample_probes!(G, Ξ, R::BlocksOfDiagonals, rng)
    d = nblocks(R)
    draw_standard_probes!(Ξ, rng)
    @inbounds for j in axes(G, 2), (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
        s = zero(eltype(G))
        for l in axes(Ri, 1)
            s += Ri[l, k] * Ξ[(l-1)*d+i, j]
        end
        G[(k-1)*d+i, j] = s
    end
    return G
end

"""
    estimate_diagonal!(out, H, R::BlocksOfDiagonals, n)

The diagonal of ``H R^T R H^T``, estimated as the mean of ``(H g)^2`` over `n` probes
``g = R^T ξ``. Only ``E_0 g`` and ``E_1 g`` are formed, since ``H`` reads nothing else.
"""
function estimate_diagonal!(out, H::MatrixFreeObservationOperator, R::BlocksOfDiagonals, n)
    (; ξ, e0, a, Jv) = H.cache
    rng = integrator_rng(H.integ)
    d = length(out)
    fill!(out, 0)
    for _ in 1:n
        draw_standard_probes!(ξ, rng)
        @inbounds for (i, Ri) in enumerate(blocks(R))
            g0, g1 = zero(eltype(out)), zero(eltype(out))
            for l in axes(Ri, 1)
                g0 += Ri[l, 1] * ξ[(l-1)*d+i]
                g1 += Ri[l, 2] * ξ[(l-1)*d+i]
            end
            e0[i], a[i] = g0, g1
        end
        jvp!(Jv, H, e0)
        @.. out = out + (a - Jv)^2
    end
    @.. out = out / n
    return out
end

"""
    jacobi_preconditioner!(P, S::MatrixFreeObservedCovariance)

The diagonal `P` of a Jacobi preconditioner for ``S = H Σ H^T``: ``\\operatorname{diag}(S)``
as estimated when `S` was built, or, with a diagonal Jacobian ``J_D``, the diagonal of
``H_D Σ H_D^T`` for ``H_D = E_1 - J_D E_0``. Its entries are raised to at least `eps` times
the largest one, so that they are positive.
"""
function jacobi_preconditioner!(P, S::MatrixFreeObservedCovariance)
    (; H, ΣE₀ᵀ, ΣE₁ᵀ) = S.C
    J = H.cache.diagonal_jacobian
    if isnothing(J)
        copy!(P, S.diagonal)
    else
        d = length(P)
        # Var(uᵢ' - Jᵢᵢ uᵢ) = Σ₁₁ - 2 Jᵢᵢ Σ₀₁ + Jᵢᵢ² Σ₀₀
        @views P .= ΣE₁ᵀ[(d+1):2d] .- 2 .* J .* ΣE₁ᵀ[1:d] .+ J .^ 2 .* ΣE₀ᵀ[1:d]
    end
    lower = max(floatmin(eltype(P)), eps(eltype(P)) * maximum(P))
    @.. P = max(P, lower)
    return P
end

# The tolerance of the linear solves: the solver's `reltol`, capped at 1e-2
function krylov_rtol(H::MatrixFreeObservationOperator)
    T = eltype(H.u)
    return min(T(1e-2), T(minimum(H.integ.opts.reltol)))
end

struct LinearSolveFailure <: Exception end

"""
    krylov_solve!(S, b, P, rtol, part)

``S^{-1} b`` to the relative tolerance `rtol`, with the Jacobi preconditioner
`Diagonal(P)`: by CG for a vector `b`, and for the columns of a matrix `b` jointly by
block-GMRES, or by CG one by one if there are more than half as many columns as rows. It
counts in the statistics of `part`, `:mean`, `:covariance` or `:calibration`, and throws a
`LinearSolveFailure` if it does not converge.
"""
function krylov_solve!(
    S::MatrixFreeObservedCovariance, b::AbstractVector, P, rtol, part;
    itmax=MAX_CG_ITERATIONS,
)
    H = S.C.H
    workspace = H.cache.cg
    cg!(workspace, S, b; M=Diagonal(P), ldiv=true, atol=zero(rtol), rtol, itmax)
    count_solve!(H, part, Krylov.iteration_count(workspace))
    Krylov.issolved(workspace) || throw(LinearSolveFailure())
    return Krylov.solution(workspace)
end
function krylov_solve!(S::MatrixFreeObservedCovariance, B::AbstractMatrix, P, rtol, part)
    H = S.C.H
    c = H.cache
    if isnothing(c.block_gmres)
        # Block-GMRES needs at least twice as many rows as right-hand sides
        for j in axes(B, 2)
            copy!(c.z, view(B, :, j))
            copy!(view(c.W, :, j), krylov_solve!(S, c.z, P, rtol, part))
        end
        return c.W
    end
    # Preconditioned on both sides, so that the residual is weighted as in the
    # preconditioned CG
    @.. c.sqrtP = sqrt(P)
    M = Diagonal(c.sqrtP)
    block_gmres!(c.block_gmres, S, B; M, N=M, ldiv=true, atol=zero(rtol), rtol,
        itmax=MAX_BLOCK_GMRES_ITERATIONS)
    count_solve!(H, part, Krylov.iteration_count(c.block_gmres))
    Krylov.issolved(c.block_gmres) || throw(LinearSolveFailure())
    return Krylov.solution(c.block_gmres)
end

# Count a linear solve for `part`, `:mean`, `:covariance` or `:calibration`
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
        stats.nsolve_calibration += 1
        stats.niter_calibration += niter
    end
    return nothing
end

"""
    MatrixFreeGain(C, S, P)

The Kalman gain ``K = Σ H^T S^{-1}``, as the cross-covariance `C` ``= Σ H^T`` and
`S` ``= H Σ H^T``, with the diagonal `P` of the Jacobi preconditioner for `S`.
`update_mean!` returns it for `update_cov!`.
"""
struct MatrixFreeGain{CT,ST,PT}
    C::CT
    S::ST
    P::PT
end

"""
    update_mean!(x_out, x_pred, obs::MatrixFreeLinearizedObservation; cache)

The mean of the update on a matrix-free observation: ``S^{-1} z`` with ``S = H Σ H^T`` by
CG with a Jacobi preconditioner, and ``μ^F = μ - Σ H^T S^{-1} z``. It returns the gain
``K = Σ H^T S^{-1}`` as a `MatrixFreeGain`, and the log-likelihood `NaN`.
"""
function update_mean!(x_out, x_pred, obs::MatrixFreeLinearizedObservation; cache)
    (; m, H) = obs
    c = H.cache
    S = MatrixFreeObservedCovariance(
        H, x_pred.Σ.R, c.HΣHᵀ; estimate_diagonal=isnothing(c.diagonal_jacobian))
    P = jacobi_preconditioner!(c.P, S)
    z = if m === x_pred.μ
        obs.r
    else
        # r + H (μ - m), with m - μ in x_out.μ
        Δ = x_out.μ .= m .- x_pred.μ
        mul!(c.z, H, Δ)
        @.. c.z = obs.r - c.z
    end
    w = krylov_solve!(S, z, P, krylov_rtol(H), :mean)
    ztSinvz = dot(z, w)
    mul!(x_out.μ, S.C, w)
    @.. x_out.μ = x_pred.μ - x_out.μ
    K = MatrixFreeGain(S.C, S, P)
    return (; loglikelihood=oftype(ztSinvz, NaN), ztSinvz, S, z, K, B=nothing)
end

"""
    update_cov!(x_out, x_pred, obs::MatrixFreeLinearizedObservation, K, B; cache)

The covariance of the update on a matrix-free observation, block-diagonal. Probes
``g_j \\sim \\mathcal{N}(0, Σ)`` give the samples
``(I - K H) g_j \\sim \\mathcal{N}(0, Σ^F)``, with ``K H g_j = Σ H^T S^{-1} H g_j`` and the
systems for all ``j`` solved jointly. The square root of ``Σ^F`` is that of the
block-diagonal part of their sample covariance.
"""
function update_cov!(
    x_out, x_pred, obs::MatrixFreeLinearizedObservation, K::MatrixFreeGain, B; cache,
)
    H = obs.H
    c = H.cache
    (; G, HG, gtmp) = c
    sample_probes!(G, c.Ξ, x_pred.Σ.R, integrator_rng(H.integ))
    mul!(HG, H, G)
    W = krylov_solve!(K.S, HG, K.P, krylov_rtol(H), :covariance)
    for j in axes(G, 2)
        mul!(gtmp, K.C, view(W, :, j))
        view(G, :, j) .-= gtmp
    end
    block_diagonal_sample_covariance!(x_out.Σ.R, G, (c.Y, c.M))
    return x_out
end

"""
    block_diagonal_sample_covariance!(R::BlocksOfDiagonals, G, (Y, M))

The square root `R` of the block-diagonal part of the sample covariance
``\\frac{1}{m} G G^T`` of the `m` columns of `G`, with the buffers `Y` and `M`. Each block of
`R` is computed from the rows of `G` that belong to it, `i:d:D` in the derivative-major
order of the state; the covariances between the blocks are dropped.
"""
function block_diagonal_sample_covariance!(R::BlocksOfDiagonals, G, (Y, M))
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

# The error estimate needs the diagonal of H Q(h) Hᵀ with adaptive steps, and so does the
# preconditioner of the calibration without a diagonal Jacobian
function observed_process_noise!(
    cache::AbstractODEFilterCache, obs::MatrixFreeLinearizedObservation,
)
    H = obs.H
    estimate_diagonal = H.integ.opts.adaptive || isnothing(H.cache.diagonal_jacobian)
    return MatrixFreeObservedCovariance(H, cache.Qh.R, H.cache.HQHᵀ; estimate_diagonal)
end

# zᵀ (H Q(h) Hᵀ)⁻¹ z, for the calibration, by CG with a Jacobi preconditioner
function invquad(z, S::MatrixFreeObservedCovariance; kwargs...)
    P = jacobi_preconditioner!(S.C.H.cache.PQ, S)
    return dot(z, krylov_solve!(S, z, P, eltype(S)(CALIBRATION_RTOL), :calibration))
end

diag!(v::AbstractVector, S::MatrixFreeObservedCovariance) = copy!(v, S.diagonal)
