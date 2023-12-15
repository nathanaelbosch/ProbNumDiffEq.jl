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

In this package, the LTISDE is mostly used to implement the discretization of the [`IOUP`](@ref) prior.
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


discretize(sde::LTISDE, dt::Real) =
    matrix_fraction_decomposition(drift(sde), dispersion(sde), dt)

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

function discretize_sqrt_with_quadraturetrick(sde::LTISDE, dt::Real)
    F, L = drift(sde), dispersion(sde)

    D = size(F, 1)
    d = size(L, 2)
    N = Int(D / d)
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
