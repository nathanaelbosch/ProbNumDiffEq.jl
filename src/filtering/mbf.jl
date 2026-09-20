# Preallocated scratch buffers for the backward step, created once per `smooth_solution!`
# call and reused by every step; shared by all three structural paths (dense/EK1,
# Kronecker/EK0, block-diagonal/DiagonalEK1) and sized by (D, d) = (state dimension,
# measurement dimension).
Base.@kwdef struct _MBFScratch{T,Tw<:AbstractVecOrMat{T}}
    U_Λ_prec::Matrix{T}
    U_Λ_pred::Matrix{T}
    U_Λ_new::Matrix{T}
    Stack::Matrix{T}
    R_M::Matrix{T}
    R_f::Matrix{T}
    U_ΛK::Matrix{T}
    H_prec::Matrix{T}
    K_prec::Matrix{T}
    λ_prec::Vector{T}
    λ_pred::Vector{T}
    λ_new::Vector{T}
    w::Tw
    z_sol::Tw
    v_h::Vector{T}
    w_h::Vector{T}
    tau::Vector{T}
    S_U_T::Matrix{T}
end

# `wsize` is the shape of the measurement-space work vectors `w`/`z_sol`: `(d,)` for the
# dense and block-diagonal paths, and `(d, d_ode)` for EK0's Kronecker path, which batches
# the update over the `d_ode` ODE dimensions.
function _MBFScratch(D::Integer, d::Integer, ::Type{T}; wsize=(d,)) where {T}
    return _MBFScratch(;
        U_Λ_prec=Matrix{T}(undef, D, D),
        U_Λ_pred=Matrix{T}(undef, D, D),
        U_Λ_new=Matrix{T}(undef, D, D),
        Stack=Matrix{T}(undef, D + d, D),
        R_M=Matrix{T}(undef, D, D),
        R_f=Matrix{T}(undef, D, D),
        U_ΛK=Matrix{T}(undef, D, d),
        H_prec=Matrix{T}(undef, d, D),
        K_prec=Matrix{T}(undef, D, d),
        λ_prec=zeros(T, D),
        λ_pred=zeros(T, D),
        λ_new=zeros(T, D),
        w=zeros(T, wsize...),
        z_sol=zeros(T, wsize...),
        v_h=zeros(T, D),
        w_h=zeros(T, D),
        tau=zeros(T, D),
        S_U_T=Matrix{T}(undef, d, d),
    )
end

# Allocate the scratch matching the state covariance's structure, dispatching the same way
# `_mbf_backward_step!` does instead of branching on concrete types at the call site.
_mbf_scratch(U_Λ::Matrix, cache) =
    _MBFScratch(size(U_Λ, 1), size(cache.H, 1), eltype(U_Λ))
_mbf_scratch(U_Λ::IsometricKroneckerProduct, cache) =
    _MBFScratch(
        size(U_Λ.B, 1), size(cache.H.B, 1), eltype(U_Λ);
        wsize=(size(cache.H.B, 1), U_Λ.rdim))
_mbf_scratch(U_Λ::BlocksOfDiagonals, cache) =
    _MBFScratch(
        size(blocks(U_Λ)[1], 1), size(blocks(cache.H)[1], 1), eltype(U_Λ))

"""
    _mbf_backward_step!(x_smooth_prev, λ, U_Λ, x_filt_prev, ss, P, PI, A, sc)

One backward step of the √MBF recursion: incorporate the measurement info in `ss` (or just
predict, if `ss === nothing`) into the adjoint state `(λ, U_Λ)`, then recover the smoothed
mean/covariance into `x_smooth_prev` using the filtered state `x_filt_prev`. Returns the
updated `(λ, U_Λ)` for use at the next (earlier) step.

The measurement quantities (`z`, `H`, `K`, `S_U`) were pre-computed and stored by the
forward pass in `ss` (see [`SmootherState`](@ref)), so this function does not need to
recompute the predicted covariance or the Kalman gain.

**AD limitation**: under forward-mode AD the smoothed *values* computed here are correct,
but the partials of the smoothed *covariance* (`x_smooth_prev.Σ`) can be `NaN`; smoothed
means are unaffected. Use `smoother=:rts` if you need gradients through smoothed
covariances. The full explanation lives at [`_hyperbolic_qr!`](@ref), the source of the
conditioning problem.

Dispatches on the structure of `P` (which matches that of the state covariances) so that EK0's
and DiagonalEK1's efficient Kronecker/block-diagonal representations are preserved: the λ/Λ
recursion and covariance recovery only ever touch the small `(q+1)×(q+1)` blocks (batched over
the `d` ODE dimensions for EK0, or looped independently for DiagonalEK1), never a dense `D×D`
matrix -- without this, backward smoothing would cost `O(D^3) = O((d(q+1))^3)` instead of the
`O(d)`-ish cost the rest of the package achieves for these algorithms.

`sc` is a [`_MBFScratch`](@ref) holding preallocated buffers, created once by
[`_smooth_solution_mbf!`](@ref) and reused across all backward steps.

The λ/Λ update and predict steps involve matrix products (H, K, the transition matrix) applied
repeatedly across many steps; doing this in physical coordinates re-introduces the same
ill-conditioning that motivated preconditioning elsewhere in this codebase (IWP covariances
span many orders of magnitude across derivative orders). So: precondition into this specific
transition's local coordinates (`P`, `PI` depend on the step size and are only valid for this
one transition), do the update+predict there using the h-independent preconditioned transition
`A`, then unprecondition back to physical before recovering `x_smooth_prev` below (using this
transition's own `P`/`PI`, the natural local scale for `x_filt_prev` itself -- a scale borrowed
from an adjacent transition was found empirically to reintroduce significant error).
"""

# The dense (EK1) implementation. Given a matching `scratch` the only remaining
# allocation per step is the LAPACK workspace inside `_positive_qr_r!`.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ, x_filt_prev, ss,
    P::Diagonal, PI::Diagonal, A::Matrix, sc::_MBFScratch,
)
    D = length(λ)

    # Precondition into this transition's local coordinates.
    _matmul!(sc.λ_prec, transpose(PI), λ)
    _matmul!(sc.U_Λ_prec, U_Λ, PI)

    # Incorporate this step's measurement, if there is one. Without it (`ss === nothing`,
    # a degenerate step) the recursion is a pure predict and the preconditioned λ/Λ pass
    # through unchanged.
    if !isnothing(ss)
        _matmul!(sc.H_prec, ss.H, transpose(PI))
        _matmul!(sc.K_prec, P, ss.K)
        _sqrt_mbf_update!(
            sc.λ_prec, sc.U_Λ_prec, sc.K_prec, ss.S_U, ss.z, sc.H_prec, D, sc,
        )
    end

    # Predict (still in preconditioned coordinates) and unprecondition: the physical
    # transition is Φ = PI A P, so λ_new = Φ'λ = P A' PI λ and U_Λ_new = U_Λ Φ.
    _matmul!(sc.λ_pred, transpose(A), sc.λ_prec)
    _matmul!(sc.λ_new, transpose(P), sc.λ_pred)
    _matmul!(sc.U_Λ_pred, sc.U_Λ_prec, A)
    _matmul!(sc.U_Λ_new, sc.U_Λ_pred, P)
    λ_new, U_Λ_new = sc.λ_new, sc.U_Λ_new

    # Recover smoothed mean/covariance using the now-updated λ, Λ (information from
    # measurements {i, ..., n}).
    # `U_f` is only ever read below, so alias the filtered factor instead of copying it.
    U_f = x_filt_prev.Σ.R
    _mbf_recover_mean!(x_smooth_prev.μ, U_f, λ_new, sc)
    # The recovery is done directly in physical coordinates: `V = U_Λ_new * U_f'` and
    # `R_f = R_M * U_f` (no P/PI scalings -- they would cancel exactly, since
    # `PI == P^{-1}`). This saves two D×D products per step.
    _matmul!(sc.U_Λ_pred, U_Λ_new, transpose(U_f))
    _hyperbolic_qr!(sc.R_M, sc.v_h, sc.w_h, sc.U_Λ_pred)
    _matmul!(sc.R_f, sc.R_M, U_f)
    # `sc.R_f` is a valid square-root factor of the smoothed covariance
    # (`sc.R_f'sc.R_f = Σ_s`) but generally a full (non-triangular) matrix. That is fine:
    # the package does not assume upper-triangular factors in general (e.g. `update!`
    # stores `Σ_pred.R * (I - K*H)'`), and all `PSDMatrix` operations (`Matrix`, `det`,
    # `logabsdet`, `diag`, QR stacks, ...) only rely on `R'R`. So we store the factor
    # as-is instead of paying for a per-step re-triangularizing QR.
    copy!(x_smooth_prev.Σ.R, sc.R_f)

    return λ_new, U_Λ_new
end

# Kronecker version (EK0): reduce to the shared small `.B` block, batched over the `d` ODE
# dimensions via `(q+1)×d`-shaped λ, instead of densifying to `D×D`.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ::IsometricKroneckerProduct, x_filt_prev, ss,
    P::IsometricKroneckerProduct, PI::IsometricKroneckerProduct,
    A::IsometricKroneckerProduct, sc::_MBFScratch,
)
    d = P.rdim
    Q = size(P.B, 1)
    _λ = reshape_no_alloc(λ, d, Q)'
    P_B, PI_B, A_B, U_Λ_B = P.B, PI.B, A.B, U_Λ.B

    # Precondition into this transition's local coordinates. Both are fresh matrices, so
    # `_sqrt_mbf_update!` can update them in place.
    λ_prec = PI_B' * _λ
    U_Λ_prec = U_Λ_B * PI_B

    if !isnothing(ss)
        _sqrt_mbf_update!(
            λ_prec, U_Λ_prec, P_B * ss.K.B, ss.S_U.B,
            reshape_no_alloc(ss.z, d, 1)', ss.H.B * PI_B, Q, sc)
    end

    # Predict (still in preconditioned coordinates) and unprecondition.
    λ_new_B = P_B' * (A_B' * λ_prec)
    U_Λ_new_B = (U_Λ_prec * A_B) * P_B

    U_f_B = x_filt_prev.Σ.R.B
    μ_smooth_view = reshape_no_alloc(x_smooth_prev.μ, d, Q)'
    _mbf_recover_mean!(μ_smooth_view, U_f_B, λ_new_B)
    # Same as above: recover directly in physical coordinates -- the P/PI scalings cancel
    # exactly, so `U_Λ_B * PI_B` and `U_f_B * P_B'` (and the trailing `* PI_B'`) are not needed.
    new_R_B = _mbf_recover_cov(x_filt_prev.Σ.R.B, U_Λ_new_B)
    copy!(x_smooth_prev.Σ.R.B, new_R_B)

    λ_new = similar(λ)
    reshape_no_alloc(λ_new, d, Q)' .= λ_new_B
    return λ_new, IsometricKroneckerProduct(d, U_Λ_new_B)
end

# Block-diagonal version (DiagonalEK1): unlike EK0, each of the `d` ODE dimensions has its own
# measurement Jacobian (and hence its own Kalman gain), so there is no shared small block to
# batch over. Instead, reduce to `d` independent `(q+1)`-sized problems, one per diagonal
# block, and recurse -- each recursive call dispatches to the dense method above.
function _mbf_backward_step!(
    x_smooth_prev, λ, U_Λ::BlocksOfDiagonals, x_filt_prev, ss,
    P::BlocksOfDiagonals, PI::BlocksOfDiagonals, A::BlocksOfDiagonals,
    sc::_MBFScratch,
)
    d = length(blocks(P))
    D = length(λ)
    λ_new = similar(λ)
    new_U_Λ_blocks = similar(blocks(U_Λ))

    for bi in 1:d
        _ss =
            isnothing(ss) ? nothing :
            SmootherState(
                view(ss.z, bi:bi), ss.H.blocks[bi],
                ss.K.blocks[bi], ss.S_U.blocks[bi], nothing,
            )
        _x_filt_prev =
            Gaussian(view(x_filt_prev.μ, bi:d:D), PSDMatrix(x_filt_prev.Σ.R.blocks[bi]))
        _x_smooth_prev =
            Gaussian(view(x_smooth_prev.μ, bi:d:D), PSDMatrix(x_smooth_prev.Σ.R.blocks[bi]))

        λ_bi_new, U_Λ_bi_new = _mbf_backward_step!(
            _x_smooth_prev, view(λ, bi:d:D), U_Λ.blocks[bi],
            _x_filt_prev, _ss, P.blocks[bi], PI.blocks[bi], A.blocks[bi], sc)

        λ_new[bi:d:D] .= λ_bi_new
        new_U_Λ_blocks[bi] = copy(U_Λ_bi_new)
    end
    return λ_new, BlocksOfDiagonals(new_U_Λ_blocks)
end

"""
    _mbf_recover_mean!(μ_smooth, U_f, λ, [sc])

Recover the smoothed mean in place from the filtered mean (already in `μ_smooth`), the
filtered covariance factor `U_f` (upper triangular with `Σ_filt = U_f'U_f`) and the MBF
adjoint mean `λ`, i.e. `μ_smooth -= Σ_filt * λ = U_f' * (U_f * λ)`.

Passing a [`_MBFScratch`](@ref) `sc` runs the two products through its preallocated
length-`D` buffers, which the dense path does to keep the backward step allocation-free.
The Kronecker path batches over the ODE dimensions, so its `λ` is a `(q+1)×d` matrix that
does not fit those buffers and it uses the allocating method instead.
"""
function _mbf_recover_mean!(μ_smooth, U_f::Matrix, λ)
    μ_smooth .-= U_f' * (U_f * λ)
end
function _mbf_recover_mean!(μ_smooth, U_f, λ, sc::_MBFScratch)
    _matmul!(sc.λ_prec, U_f, λ)
    _matmul!(sc.λ_pred, transpose(U_f), sc.λ_prec)
    μ_smooth .-= sc.λ_pred
    return μ_smooth
end

"""
    _mbf_recover_cov(U_f, U_Λ)

Recover the smoothed covariance square-root factor from the filtered covariance
(`U_f`, upper triangular such that `Σ_filt = U_f'U_f`) and the MBF adjoint
information matrix (`U_Λ`, such that `Λ = U_Λ'U_Λ`), i.e. `R` with
`R'R = Σ_filt - Σ_filt*Λ*Σ_filt`. Writing `Σ_smooth = U_f' * (I - W*W') * U_f` with
`W = U_f*U_Λ'`, this is computed via a hyperbolic QR factorization of `V = W'`, which
guarantees a PSD result (unlike forming `Σ_filt - Σ_filt*Λ*Σ_filt` densely, which
squares the condition number and can catastrophically cancel when the smoothed
covariance is much smaller than the filtered one). `U_f` is a factor of the filtered
covariance and `U_Λ` of the adjoint information matrix, both in the same coordinates.
"""
function _mbf_recover_cov(U_f::Matrix, U_Λ::Matrix)
    V = U_Λ * U_f'
    R_M = _hyperbolic_qr!(V)
    return R_M * U_f
end

function _hyperbolic_qr!(V::Matrix{T}) where {T}
    d = size(V, 1)
    return _hyperbolic_qr!(
        Matrix{T}(I, d, d), Vector{T}(undef, d), Vector{T}(undef, d), V,
    )
end

# In-place variant of [`_hyperbolic_qr!`](@ref) using the preallocated output matrix `top`
# (initialized to I here) and work vectors `v_h`, `w_h`; destroys `V`.
#
# AD limitation (this is the one authoritative discussion: the `_mbf_backward_step!`
# docstring and `test/autodiff.jl` point here, and the EK0/EK1/DiagonalEK1 docstrings and
# the docs' "Smoothing and automatic differentiation" section carry only the user-facing
# one-liner and the `smoother=:rts` workaround). Under forward-mode AD (ForwardDiff.Dual
# eltypes) the *values* computed by the MBF smoother are correct and the smoothed means are
# unaffected, but the partials of the smoothed *covariances* can be NaN. The cause sits
# here: `β = 1 / (α * (α - top[i, i]))` below is ~1/α²-conditioned in the column's J-norm
# `α = sqrt(max(asq - bsq, 0))`, and as `α → 0⁺` (a nearly singular `I - V'V`, i.e. a
# smoothed covariance much smaller than the filtered one, and short of the
# `α < eps(T) * d` degenerate-column cutoff) the partials of `α`, and hence of `β` and `τ`,
# diverge and then cancel catastrophically in the recovered covariance downstream. The
# divergence was confirmed analytically to set in well before the cutoff is reached, i.e.
# it is an intrinsic conditioning property of hyperbolic rotations (Gibbs 2011) rather than
# an artifact of the `max(·, 0)` clamp's kink or a bug in this implementation, and no fix
# is known that preserves the Float64 accuracy of the degenerate-column handling. Users who
# need gradients through smoothed covariances should pass `smoother=:rts`; the limitation
# is pinned by the `@test_broken` pair in `test/autodiff.jl`.
function _hyperbolic_qr!(
    top::Matrix{T}, v_h::Vector{T}, w_h::Vector{T}, V::Matrix{T},
) where {T}
    d = size(V, 1)
    fill!(top, zero(T))
    @inbounds for i in 1:d
        top[i, i] = one(T)
    end
    bot = V
    @inbounds for i in 1:d
        na = d - i + 1

        asq = zero(T)
        for l in i:d
            asq += top[l, i] * top[l, i]
        end
        bsq = zero(T)
        for l in 1:d
            bsq += bot[l, i] * bot[l, i]
        end

        α² = max(asq - bsq, zero(T))
        α = sqrt(α²)

        if α < eps(T) * d
            for l in i:d
                top[l, i] = zero(T)
            end
            for l in 1:d
                bot[l, i] = zero(T)
            end
            continue
        end

        α = top[i, i] >= 0 ? -α : α
        β = one(T) / (α * (α - top[i, i]))

        v_h[1] = top[i, i] - α
        for l in 2:na
            v_h[l] = top[i+l-1, i]
        end
        for l in 1:d
            w_h[l] = bot[l, i]
        end

        for j in (i+1):d
            τ = zero(T)
            for l in 1:na
                τ += v_h[l] * top[i+l-1, j]
            end
            for l in 1:d
                τ -= w_h[l] * bot[l, j]
            end
            τ *= β

            for l in 1:na
                top[i+l-1, j] -= τ * v_h[l]
            end
            for l in 1:d
                bot[l, j] -= τ * w_h[l]
            end
        end

        top[i, i] = α
        for l in (i+1):d
            top[l, i] = zero(T)
        end
        for l in 1:d
            bot[l, i] = zero(T)
        end
    end
    return top
end

"""
    _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D, sc)

Information-form update of the MBF adjoint state `(λ, U_Λ)` with the measurement quantities
of one step (`K`, `S_U`, `z`, `H`), writing into the preallocated buffers of `sc`
(a [`_MBFScratch`](@ref)).

λ update (BK-free form): with `z_code = h(m_pred)` and innovation `-z_code`,
`λ ← λ - H' (K'λ - S⁻¹ z_code)`.

Λ update (sqrt form via QR of the stack `[U_Λ * (I - K*H); M]`): the second stack block must
be a factor `M` with `M'M = S⁻¹`. With the upper triangular Cholesky factor `S_U`
(`S = S_U'S_U`) that is `M = S_U⁻ᵀ H`, since `(S_U⁻ᵀ H)'(S_U⁻ᵀ H) = H'S⁻¹H` (the lower
triangular solve with `S_U'` turns `S_U` into an inverted *lower* triangular factor);
instead using `M = S_U⁻¹ H` would give `H'(S_U S_U')⁻¹H ≠ H'S⁻¹H` (unless `S_U` is
diagonal) and silently corrupt Λ.
"""
function _sqrt_mbf_update!(λ, U_Λ, K, S_U, z, H, D, sc::_MBFScratch)
    # `S_U_T = S_U'` as a contiguous matrix so that the triangular solves below hit BLAS.
    copyto!(sc.S_U_T, transpose(S_U))

    # λ update (BK-free form); see the docstring for the sign conventions
    _matmul!(sc.w, transpose(K), λ)
    copyto!(sc.z_sol, z)
    ldiv!(LowerTriangular(sc.S_U_T), sc.z_sol)
    ldiv!(UpperTriangular(S_U), sc.z_sol)
    sc.w .-= sc.z_sol
    _matmul!(λ, transpose(H), sc.w, -1.0, 1.0)

    # Λ update (sqrt form via QR of the stack; see the docstring for the S_U⁻ᵀH factor).
    # The first stack row `U_Λ * (I - K*H)` is computed as `U_Λ - (U_Λ*K)*H` to avoid
    # materializing the D×D matrix `BK`.
    _matmul!(sc.U_ΛK, U_Λ, K)
    StackTop = view(sc.Stack, 1:D, :)
    copyto!(StackTop, U_Λ)
    _matmul!(StackTop, sc.U_ΛK, H, -1.0, 1.0)
    StackBot = view(sc.Stack, (D+1):(D+size(H, 1)), :)
    copyto!(StackBot, H)
    ldiv!(LowerTriangular(sc.S_U_T), StackBot)
    _positive_qr_r!(U_Λ, sc.Stack, sc.tau)
    return nothing
end

"""
    _extract_measurement_chol(measurement_chol)

Take ownership of the upper triangular Cholesky factor of the measurement covariance that
`update!` wrote into `cache.measurement_chol` (see the `S_chol_out` argument of
[`update!`](@ref)), by copying it out of the cache buffer that the next step overwrites.
"""
_extract_measurement_chol(measurement_chol) = copy(measurement_chol)

"""
    _measurement_chol_scale(mle_diffusion::Diagonal) -> Number or Diagonal

Pre-compute the measurement-Cholesky scale factor `M = sqrt.(mle_diffusion)` once, so
that the per-step [`_rescale_measurement_chol!`](@ref) loop avoids redundant allocations.
Returns a scalar for isotropic (FillArrays-backed) diffusions, a `Diagonal` otherwise.
"""
_measurement_chol_scale(D::Diagonal{<:Number,<:FillArrays.Fill}) = sqrt(D.diag.value)
_measurement_chol_scale(D::Diagonal) = Diagonal(sqrt.(D.diag))

"""
    _rescale_measurement_chol!(S_U, scale)

Rescale a stored smoother-state measurement-covariance Cholesky factor `S_U` after the
solution's covariances were calibrated (see [`calibrate_solution!`](@ref)). `scale` is
the pre-computed output of [`_measurement_chol_scale`](@ref): a scalar for isotropic
diffusions, a `Diagonal` for per-dimension ones (where the measurement Jacobians of
EK0 and DiagonalEK1 are dimension-pure). The calibrated factor is `S_U' = S_U * scale`.
"""
# A scalar scale is just `rmul!`, for which every covariance structure already has a
# method (`Matrix`, `IsometricKroneckerProduct`, `BlocksOfDiagonals`). So is a `Diagonal`
# scale on a dense factor; only the block-diagonal case needs to pair blocks with entries.
_rescale_measurement_chol!(S_U, scale) = (rmul!(S_U, scale); S_U)
function _rescale_measurement_chol!(S_U::BlocksOfDiagonals, M::Diagonal)
    @simd ivdep for i in eachindex(blocks(S_U))
        rmul!(blocks(S_U)[i], M.diag[i])
    end
    return S_U
end
function _rescale_measurement_chol!(S_U::IsometricKroneckerProduct, M::Diagonal)
    throw(
        ArgumentError(
            "Per-dimension diffusion calibration is not supported with isometric Kronecker covariances.",
        ),
    )
end

# Factorize `Stack` (destroying it) and write its R factor, normalized to a positive
# diagonal, into `R_dest` (zeros below the diagonal).
function _positive_qr_r!(R_dest::Matrix, Stack::Matrix, tau::Vector)
    D = size(Stack, 2)
    if eltype(Stack) <: LinearAlgebra.BlasFloat
        LinearAlgebra.LAPACK.geqrf!(Stack, tau)
    else
        # Generic fallback for e.g. ForwardDiff.Dual eltypes (allocating, but AD rarely
        # smooths large systems).
        Stack[1:D, 1:D] .= qr(Stack).R
    end
    @inbounds for i in 1:D
        if Stack[i, i] < 0
            @views Stack[i, 1:D] .*= -1
        end
    end
    @inbounds for j in 1:D, i in 1:D
        R_dest[i, j] = i <= j ? Stack[i, j] : zero(eltype(R_dest))
    end
    return R_dest
end
