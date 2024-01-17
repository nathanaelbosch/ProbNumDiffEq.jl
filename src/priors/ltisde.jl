"""
    LTISDE(F::AbstractMatrix, L::AbstractMatrix)

Linear time-invariant stochastic differential equation.

A LTI-SDE is a stochastic differential equation of the form
```math
dX_t = F X_t dt + L dW_t
```
where ``X_t`` is the state, ``W_t`` is a Wiener process, and ``F`` and ``L`` are matrices.
This `LTISDE` object holds the matrices ``F`` and ``L``.
It also provides some functionality to discretize the SDE via a matrix-fraction decomposition.
See: [`discretize(::LTISDE, ::Real)`](@ref).
"""
struct LTISDE{AT<:AbstractMatrix,BT<:AbstractVecOrMat}
    F::AT
    L::BT
end
drift(sde::LTISDE) = sde.F
dispersion(sde::LTISDE) = sde.L

iterate(sde::LTISDE) = sde.F, true
iterate(sde::LTISDE, s) = s ? (sde.L, false) : nothing
length(sde::LTISDE) = 2

"""
    discretize(p::LTISDE, step_size::Real)

Compute the transition matrices of the SDE solution for a given step size.
"""
discretize(sde::LTISDE, dt::Real) = discretize(drift(sde), dispersion(sde), dt)
discretize(F::AbstractMatrix, L::AbstractMatrix, dt::Real) = begin
    method = FiniteHorizonGramians.ExpAndGram{eltype(F),13}()
    A, QR = FiniteHorizonGramians.exp_and_gram_chol(F, L, dt, method)
    Q = PSDMatrix(QR)
    return A, Q
end
discretize(F::IsometricKroneckerProduct, L::IsometricKroneckerProduct, dt::Real) = begin
    method = FiniteHorizonGramians.ExpAndGram{eltype(F.B),13}()
    A_breve, QR_breve = FiniteHorizonGramians.exp_and_gram_chol(F.B, L.B, dt, method)
    A = IsometricKroneckerProduct(F.ldim, A_breve)
    Q = PSDMatrix(IsometricKroneckerProduct(F.ldim, QR_breve))
    return A, Q
end

function matrix_fraction_decomposition(
    drift::IsometricKroneckerProduct,
    dispersion::IsometricKroneckerProduct,
    dt::Real,
)
    d = drift.ldim
    A_breve, Q_breve = matrix_fraction_decomposition(drift.B, dispersion.B, dt)
    return IsometricKroneckerProduct(d, A_breve), IsometricKroneckerProduct(d, Q_breve)
end

function matrix_fraction_decomposition(
    drift::AbstractMatrix,
    dispersion::AbstractVecOrMat,
    dt::Real,
)
    d = size(drift, 1)
    M = [drift dispersion*dispersion'; zero(drift) -drift']
    Mexp = exponential!(dt * M)
    A = Mexp[1:d, 1:d]
    Q = Mexp[1:d, d+1:end] * A'
    return A, Q
end

# Previous implementation, outdated thanks to FiniteHorizonGramians.jl:
function _discretize_sqrt_with_quadraturetrick(sde::LTISDE, dt::Real)
    F, L = drift(sde), dispersion(sde)

    D = size(F, 1)
    d = size(L, 2)
    N = D # more robust than Int(D / d)
    R = similar(F, N * d, D)
    method = ExpMethodHigham2005()
    expcache = ExponentialUtilities.alloc_mem(F, method)

    Ah = exponential!(dt * F, method, expcache)

    chol_integrand(τ) = begin
        E = exponential!((dt - τ) * F', method, expcache)
        L'E
    end
    nodes, weights = gausslegendre(N)
    b, a = dt, 0
    @. nodes = (b - a) / 2 * nodes + (a + b) / 2
    @. weights = (b - a) / 2 * weights
    @simd ivdep for i in 1:N
        R[(i-1)*d+1:i*d, 1:D] .= sqrt(weights[i]) .* chol_integrand(nodes[i])
    end

    M = R'R |> Symmetric
    chol = cholesky!(M, check=false)
    Qh_R = if issuccess(chol)
        chol.U |> Matrix
    else
        qr!(R).R |> Matrix
    end

    return Ah, Qh_R
end
