# The linearization of the `MatfreeEK1` and the update on its observation: the Jacobian of
# the ODE is only accessed through `f.jvp` and `f.vjp`, the linear systems are solved with
# Krylov methods, and the posterior covariance is block-diagonal, estimated from probes.

# Probes for the diagonals of `H Σ Hᵀ` and `H Q(h) Hᵀ`; they cost one JVP each, no solve
const DIAGONAL_PROBES = 32
const CALIBRATION_RTOL = 0.1
const MAX_CG_ITERATIONS = 10_000
const MAX_BLOCK_GMRES_ITERATIONS = 200

function check_supported(::MatrixFreeJacobian; diffusionmodel, pn_observation_noise)
    unsupported = String[]
    isnothing(pn_observation_noise) || push!(unsupported, "`pn_observation_noise`")
    diffusionmodel isa Union{DynamicDiffusion,FixedDiffusion} ||
        push!(unsupported, "`diffusionmodel = $(repr(diffusionmodel))`")
    isempty(unsupported) || throw(
        ArgumentError(
            "The `MatfreeEK1` does not support $(join(unsupported, " or ")) yet."),
    )
    return nothing
end

function check_matrix_free_inputs(f)
    unsupported =
        f isa DynamicalODEFunction ? "second-order ODEs" :
        f.mass_matrix !== I ? "mass matrices" : nothing
    isnothing(unsupported) ||
        throw(ArgumentError("The `MatfreeEK1` does not support $unsupported yet."))
    missing_products = [string("`", name, "`") for name in (:jvp, :vjp)
                                if isnothing(getproperty(f, name))]
    isempty(missing_products) || throw(
        ArgumentError(
            "The `MatfreeEK1` needs the Jacobian-vector product `jvp(Jv, v, u, p, t)` " *
            "and the vector-Jacobian product `vjp(Jᵀv, v, u, p, t)` of the ODE function, " *
            "but $(join(missing_products, " and ")) is missing. Pass them as " *
            "`ODEFunction(f; jvp, vjp)`."),
    )
    return nothing
end

function make_linearization_cache(
    linearization::MatrixFreeJacobian, C::CovarianceStructure{T}, u, f,
) where {T}
    check_matrix_free_inputs(f)
    @assert C isa BlockDiagonalCovariance
    d, q = C.d, C.q
    D = d * (q + 1)
    m = something(linearization.nprobes, 2(q + 1))
    m >= q + 1 || throw(
        ArgumentError(
            "`nprobes` must be at least the number of derivatives in the " *
            "state, $(q + 1), but it is $m.",
        ),
    )
    vec() = zeros(T, d)
    has_block_gmres = 2m <= d
    return (;
        r=vec(), u=vec(),
        diagonal_jacobian=isnothing(linearization.diagonal_jacobian) ? nothing : vec(),
        failed=Ref(false),
        v=vec(), Jv=vec(), a=vec(), e0=vec(), z=vec(),
        Σcols=(zeros(T, D), zeros(T, D)), Qcols=(zeros(T, D), zeros(T, D)),
        P=vec(), sqrtP=vec(), PQ=vec(), HQH_diag=vec(), has_HQH_diag=Ref(false),
        ξ=zeros(T, D), Ξ=zeros(T, D, m), G=zeros(T, D, m), B=zeros(T, d, m),
        Y=zeros(T, m, q + 1), M=zeros(T, q + 1, q + 1), gtmp=zeros(T, D),
        cg=CgWorkspace(d, d, Vector{T}),
        block_gmres=has_block_gmres ?
                    BlockGmresWorkspace(d, d, m, Vector{T}, Matrix{T}; memory=20) : nothing,
        W=has_block_gmres ? nothing : zeros(T, d, m),
    )
end

"""
    JacobianProducts

The Jacobian ``H = E_1 - J E_0`` of the observation of the ODE at the point `u`, as an
operator: products with ``J`` and ``J^T`` call `f.jvp` and `f.vjp`. It also carries what the
update on it needs: the random number generator for the probes, the relative tolerance of
the linear solves, and the buffers of the linearization cache.
"""
struct JacobianProducts{F,U,P,TT,RNG,T,C}
    f::F
    u::U
    p::P
    t::TT
    rng::RNG
    rtol::T
    cache::C
end

const MatrixFreeObservation = LinearizedObservation{<:Any,<:Any,<:JacobianProducts}

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
    c.failed[] = false
    T = eltype(c.u)
    rtol = min(T(1e-2), T(minimum(integ.opts.reltol)))
    H = JacobianProducts(integ.f, c.u, p, t, integrator_rng(integ), rtol, c)
    return LinearizedObservation(m, c.r, H, R)
end

linear_solve_failed(obs::MatrixFreeObservation) = obs.H.cache.failed[]

# `v` is copied into a buffer first, since `f.jvp` and `f.vjp` may require a `Vector`
jvp!(out, H::JacobianProducts, v) = (H.f.jvp(out, copy!(H.cache.v, v), H.u, H.p, H.t); out)
vjp!(out, H::JacobianProducts, v) = (H.f.vjp(out, copy!(H.cache.v, v), H.u, H.p, H.t); out)

# The first two columns of the blocks of `Σ = RᵀR`, in the derivative-major order of the
# state. `H` reads only derivatives 0 and 1, so `H Σ Hᵀ` and `Σ Hᵀ` need no other entries.
function first_two_columns!((c0, c1), R::BlocksOfDiagonals)
    d = nblocks(R)
    for (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
        c0[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 1))
        c1[(k-1)*d+i] = dot(view(Ri, :, k), view(Ri, :, 2))
    end
    return (c0, c1)
end

"""
    ObservedCovariance(H::JacobianProducts, Σcols)

``H Σ H^T`` as an operator for Krylov.jl, for a block-diagonal ``Σ`` given by the first
two columns of its blocks, `Σcols`.
"""
struct ObservedCovariance{HT,C}
    H::HT
    Σcols::C
end
size(S::ObservedCovariance) = (length(S.H.u), length(S.H.u))
size(S::ObservedCovariance, i::Integer) = length(S.H.u)
eltype(S::ObservedCovariance) = eltype(S.H.u)

# H Σ Hᵀ w = (Σ₁₀ a + Σ₁₁ w) - J (Σ₀₀ a + Σ₀₁ w), with a = -Jᵀ w
function mul!(y::AbstractVector, S::ObservedCovariance, w::AbstractVector)
    (; a, e0, Jv) = S.H.cache
    c0, c1 = S.Σcols
    d = length(a)
    vjp!(a, S.H, w)
    @.. a = -a
    @views e0 .= c0[1:d] .* a .+ c1[1:d] .* w
    @views y .= c0[(d+1):2d] .* a .+ c1[(d+1):2d] .* w
    jvp!(Jv, S.H, e0)
    y .-= Jv
    return y
end
function mul!(Y::AbstractMatrix, S::ObservedCovariance, W::AbstractMatrix)
    for j in axes(W, 2)
        mul!(view(Y, :, j), S, view(W, :, j))
    end
    return Y
end

# Σ Hᵀ w = Σ (E₀ᵀ a + E₁ᵀ w), with a = -Jᵀ w
function gain_times!(out, S::ObservedCovariance, w)
    (; a) = S.H.cache
    c0, c1 = S.Σcols
    d = length(a)
    vjp!(a, S.H, w)
    @.. a = -a
    for k in 0:(length(out)÷d-1)
        idx = (k*d+1):((k+1)*d)
        @views out[idx] .= c0[idx] .* a .+ c1[idx] .* w
    end
    return out
end

# A Jacobi preconditioner needs positive entries
function floor_diagonal!(P)
    lower = max(floatmin(eltype(P)), eps(eltype(P)) * maximum(P))
    @.. P = max(P, lower)
    return P
end

# diag(H_D Σ H_Dᵀ) = Σ₁₁ - 2 J_D Σ₀₁ + J_D² Σ₀₀ for H_D = E₁ - J_D E₀ with J_D = diag(J)
function diagonal_jacobian_diag!(P, S::ObservedCovariance)
    J = S.H.cache.diagonal_jacobian
    c0, c1 = S.Σcols
    d = length(P)
    @views P .= c1[(d+1):2d] .- 2 .* J .* c1[1:d] .+ J .^ 2 .* c0[1:d]
    return floor_diagonal!(P)
end

# Columns of `G` drawn from N(0, RᵀR) with the standard normal probes `Ξ`, and B = H G
function draw_probes!(G, B, Ξ, R::BlocksOfDiagonals, H::JacobianProducts)
    d = nblocks(R)
    Random.randn!(H.rng, Ξ)
    @inbounds for j in axes(G, 2), (i, Ri) in enumerate(blocks(R)), k in axes(Ri, 2)
        s = zero(eltype(G))
        for l in axes(Ri, 1)
            s += Ri[l, k] * Ξ[(l-1)*d+i, j]
        end
        G[(k-1)*d+i, j] = s
    end
    for j in axes(G, 2)
        @views jvp!(B[:, j], H, G[1:d, j])
        @views B[:, j] .= G[(d+1):2d, j] .- B[:, j]
    end
    return G, B
end

# diag(H RᵀR Hᵀ), estimated from `n` probes H g with g ~ N(0, RᵀR)
function estimate_observed_diagonal!(out, R::BlocksOfDiagonals, H::JacobianProducts, n)
    (; ξ, e0, a, Jv) = H.cache
    d = length(out)
    fill!(out, 0)
    for _ in 1:n
        Random.randn!(H.rng, ξ)
        # E₀ g and E₁ g, for g = Rᵀ ξ
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
    return floor_diagonal!(out)
end

function cg_solve!(c, A, b, P, rtol)
    cg!(c.cg, A, b; M=Diagonal(P), ldiv=true, atol=zero(rtol), rtol,
        itmax=MAX_CG_ITERATIONS)
    Krylov.issolved(c.cg) || (c.failed[] = true)
    return Krylov.solution(c.cg)
end

# S⁻¹ B for the probes B, with a Jacobi preconditioner applied on both sides, so that the
# residual is weighted as in the preconditioned CG
function solve_probes!(c, S, rtol)
    if isnothing(c.block_gmres)
        # Block-GMRES needs at least twice as many rows as right-hand sides
        for j in axes(c.B, 2)
            copy!(c.z, view(c.B, :, j))
            copy!(view(c.W, :, j), cg_solve!(c, S, c.z, c.P, rtol))
        end
        return c.W
    else
        @.. c.sqrtP = sqrt(c.P)
        M = Diagonal(c.sqrtP)
        block_gmres!(c.block_gmres, S, c.B; M, N=M, ldiv=true, atol=zero(rtol), rtol,
            itmax=MAX_BLOCK_GMRES_ITERATIONS)
        Krylov.issolved(c.block_gmres) || (c.failed[] = true)
        return Krylov.solution(c.block_gmres)
    end
end

"""
    update_mean!(x_out, x_pred, obs::MatrixFreeObservation; cache)

The mean of the update on a matrix-free observation: ``S^{-1} z`` with ``S = H Σ H^T`` by
CG with a Jacobi preconditioner, and ``μ^F = μ - Σ H^T S^{-1} z``. Without a diagonal
Jacobian, ``\\operatorname{diag}(S)`` is estimated from probes. The log-likelihood is `NaN`.
"""
function update_mean!(x_out, x_pred, obs::MatrixFreeObservation; cache)
    (; m, H) = obs
    c = H.cache
    d = length(c.u)
    S = ObservedCovariance(H, first_two_columns!(c.Σcols, x_pred.Σ.R))
    P = if isnothing(c.diagonal_jacobian)
        estimate_observed_diagonal!(c.P, x_pred.Σ.R, H, DIAGONAL_PROBES)
    else
        diagonal_jacobian_diag!(c.P, S)
    end
    z = if m === x_pred.μ
        obs.r
    else
        # r + H (μ - m), with m - μ in x_out.μ
        Δ = x_out.μ .= m .- x_pred.μ
        jvp!(c.Jv, H, view(Δ, 1:d))
        @views c.z .= obs.r .- Δ[(d+1):2d] .+ c.Jv
    end
    w = cg_solve!(c, S, z, P, H.rtol)
    ztSinvz = dot(z, w)
    gain_times!(x_out.μ, S, w)
    @.. x_out.μ = x_pred.μ - x_out.μ
    return (; loglikelihood=oftype(ztSinvz, NaN), ztSinvz, S, z, K=nothing, B=nothing)
end

"""
    update_cov!(x_out, x_pred, obs::MatrixFreeObservation, K, B; cache)

The covariance of the update on a matrix-free observation, block-diagonal. Probes
``g_j \\sim \\mathcal{N}(0, Σ)`` give the samples ``(I - K H) g_j \\sim \\mathcal{N}(0, Σ^F)``,
with the systems ``S w_j = H g_j`` solved jointly by block-GMRES. Each block of the square
root of ``Σ^F`` is a triangular factor of the samples' entries in that block, scaled by
``1/\\sqrt{m}`` for ``m`` probes.
"""
function update_cov!(x_out, x_pred, obs::MatrixFreeObservation, K, B; cache)
    H = obs.H
    c = H.cache
    (; G, Y, M, gtmp) = c
    S = ObservedCovariance(H, c.Σcols)
    draw_probes!(G, c.B, c.Ξ, x_pred.Σ.R, H)
    W = solve_probes!(c, S, H.rtol)
    for j in axes(G, 2)
        gain_times!(gtmp, S, view(W, :, j))
        view(G, :, j) .-= gtmp
    end
    d, m = length(c.u), size(G, 2)
    for (i, Ri) in enumerate(blocks(x_out.Σ.R))
        rmul!(transpose!(Y, view(G, i:d:size(G, 1), :)), 1 / sqrt(m))
        chol = cholesky!(Symmetric(_matmul!(M, Y', Y)), check=false)
        if issuccess(chol) && is_well_conditioned(chol.U)
            copy!(Ri, chol.U)
        else
            copy!(Ri, triangularize!(Y; cachemat=nothing))
        end
    end
    return x_out
end

"""
    ObservedProcessNoise

``H Q(h) H^T`` for a matrix-free observation, from [`observed_process_noise!`](@ref), with
the square root `Q_R` of ``Q(h)``. `invquad` solves with it by preconditioned CG, for the
calibration, and `diag!` returns its diagonal estimated from probes, for the error estimate.
"""
struct ObservedProcessNoise{ST,RT}
    S::ST
    Q_R::RT
end

function observed_process_noise!(cache::AbstractODEFilterCache, obs::MatrixFreeObservation)
    H = obs.H
    S = ObservedCovariance(H, first_two_columns!(H.cache.Qcols, cache.Qh.R))
    H.cache.has_HQH_diag[] = false
    return ObservedProcessNoise(S, cache.Qh.R)
end

# The diagonal of H Q(h) Hᵀ, estimated from probes at its first use in a step: the error
# estimate needs it with adaptive steps, and the calibration without a diagonal Jacobian
function estimated_diagonal!(HQH::ObservedProcessNoise)
    c = HQH.S.H.cache
    if !c.has_HQH_diag[]
        estimate_observed_diagonal!(c.HQH_diag, HQH.Q_R, HQH.S.H, DIAGONAL_PROBES)
        c.has_HQH_diag[] = true
    end
    return c.HQH_diag
end

diag!(v::AbstractVector, HQH::ObservedProcessNoise) = copy!(v, estimated_diagonal!(HQH))

function invquad(z, HQH::ObservedProcessNoise; kwargs...)
    (; S) = HQH
    c = S.H.cache
    P =
        isnothing(c.diagonal_jacobian) ? estimated_diagonal!(HQH) :
        diagonal_jacobian_diag!(c.PQ, S)
    return dot(z, cg_solve!(c, S, z, P, oftype(S.H.rtol, CALIBRATION_RTOL)))
end
