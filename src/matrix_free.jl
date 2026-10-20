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
    linearization::MatrixFreeJacobian, C::CovarianceStructure{T}, u, f,
) where {T}
    @assert C isa BlockDiagonalCovariance
    d, q = C.d, C.q
    D = d * (q + 1)
    m = number_of_probes(linearization, q)
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
`diagonal`, which `diag!` and `approximate_diagonal!` read; without, `diagonal` holds no
valid values.
"""
struct MatrixFreeObservedCovariance{CT,RT,V} <: AbstractMatrixFreeOperator
    C::CT
    R::RT
    diagonal::V
    function MatrixFreeObservedCovariance(
        H, R::BlocksOfDiagonals, buffers; estimate_diagonal::Bool,
    )
        C = MatrixFreeCrossCovariance(H, R, buffers.ΣEᵀ)
        estimate_diagonal && estimate_diagonal!(buffers.diagonal, H, R, DIAGONAL_PROBES)
        return new{typeof(C),typeof(R),typeof(buffers.diagonal)}(C, R, buffers.diagonal)
    end
end
size(S::MatrixFreeObservedCovariance) = (length(S.diagonal), length(S.diagonal))
size(S::MatrixFreeObservedCovariance, i::Integer) = length(S.diagonal)
eltype(S::MatrixFreeObservedCovariance) = eltype(S.diagonal)

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
    approximate_diagonal!(P, S::MatrixFreeObservedCovariance)

An approximation `P` of the diagonal of ``S = H Σ H^T``, for its Jacobi preconditioner: the
probe estimate from when `S` was built, or, with the diagonal ``J_D`` of the Jacobian, the
diagonal of ``H_D Σ H_D^T`` with ``H_D = E_1 - J_D E_0``. Entries below `eps` times the
largest one are raised to it, so that all are positive.
"""
function approximate_diagonal!(P, S::MatrixFreeObservedCovariance)
    J_D = S.C.H.cache.diagonal_jacobian
    if isnothing(J_D)
        copy!(P, S.diagonal)
    else
        # (H_D Σ H_Dᵀ)ᵢᵢ = ‖√Σ H_Dᵀ eᵢ‖² = ‖r₁ - J_Dᵢ r₀‖², with the columns r₀ and r₁ of
        # the i-th block of √Σ for uᵢ and uᵢ'
        for (i, Ri) in enumerate(blocks(S.R))
            P[i] = sum(l -> (Ri[l, 2] - J_D[i] * Ri[l, 1])^2, axes(Ri, 1))
        end
    end
    lower = max(floatmin(eltype(P)), eps(eltype(P)) * maximum(P))
    @.. P = max(P, lower)
    return P
end

# The tolerance of the linear solves: the solver's `reltol`, capped at 1e-2
function krylov_rtol(H::MatrixFreeObservationOperator)
    T = eltype(H.cache.u)
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
            copy!(view(c.W, :, j), krylov_solve!(S, view(B, :, j), P, rtol, part))
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
    P = approximate_diagonal!(c.P, S)
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

The block-diagonal part of the posterior covariance, estimated from samples. This is an
approximation: the exact posterior covariance is not block-diagonal, since ``J`` couples
the blocks. The samples are probes ``g \\sim \\mathcal{N}(0, Σ)``, conditioned on the
observation as ``(I - K H) g \\sim \\mathcal{N}(0, Σ^F)``.
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
    G = sample_probes!(c.G, c.Ξ, x_pred.Σ.R, integrator_rng(H.integ))
    condition_probes!(G, H, K)
    estimate_block_diagonal_covariance!(x_out.Σ.R, G, (c.Y, c.M))
    return x_out
end

"""
    condition_probes!(G, H, K::MatrixFreeGain)

Condition the probes ``g \\sim \\mathcal{N}(0, Σ)``, the columns of `G`, on the observation:
``g - K H g`` with ``K H g = Σ H^T S^{-1} H g`` are samples of ``\\mathcal{N}(0, Σ^F)``. The
systems for all probes are solved jointly.
"""
function condition_probes!(G, H::MatrixFreeObservationOperator, K::MatrixFreeGain)
    (; HG, gtmp) = H.cache
    mul!(HG, H, G)
    W = krylov_solve!(K.S, HG, K.P, krylov_rtol(H), :covariance)
    for j in axes(G, 2)
        mul!(gtmp, K.C, view(W, :, j))
        view(G, :, j) .-= gtmp
    end
    return G
end

"""
    estimate_block_diagonal_covariance!(R::BlocksOfDiagonals, G, (Y, M))

Estimate the block-diagonal part of a covariance from samples, the `m` columns of `G`: its
square root `R` is that of the sample covariance ``\\frac{1}{m} G G^T`` without the
covariances between the blocks. Each block of `R` is computed from the rows of `G` that
belong to it, `i:d:D` in the derivative-major order of the state, with the buffers `Y` and
`M`.
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
    P = approximate_diagonal!(S.C.H.cache.PQ, S)
    return dot(z, krylov_solve!(S, z, P, eltype(S)(CALIBRATION_RTOL), :calibration))
end

diag!(v::AbstractVector, S::MatrixFreeObservedCovariance) = copy!(v, S.diagonal)
