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

function discretize_sqrt(sde::LTISDE, dt::Real)
    F, L = drift(sde), dispersion(sde)

    Ah = exp(dt * F)

    chol_integrand(τ) = L' * exp(F' * (dt - τ))
    nodes, weights = gausslegendre(10)
    b, a = dt, 0
    @. nodes = (b - a) / 2 * nodes + (a + b) / 2
    @. weights = (b - a) / 2 * weights
    mats = @. sqrt(weights) * chol_integrand(nodes)
    M = reduce(vcat, mats)
    Qh_R = qr!(M).R

    return Ah, Qh_R
end

function discretize_sqrt!(cache, sde::LTISDE, dt::Real)
    F, L = drift(sde), dispersion(sde)

    D = size(F, 1)
    d = size(L, 2)
    N = Int(D/d)
    M = similar(F, N*d, D)
    method = ExpMethodHigham2005()
    expcache = ExponentialUtilities.alloc_mem(F, method)
    @unpack C_DxD, C_dxD = cache

    Ah = exponential!(dt*F, method, expcache)

    chol_integrand(τ) = begin
        mul!(C_DxD, F', (dt - τ))
        E = exponential!(C_DxD, method, expcache)
        out = mul!(C_dxD, L', E)
    end
    nodes, weights = gausslegendre(N)
    b, a = dt, 0
    @. nodes = (b - a) / 2 * nodes + (a + b) / 2
    @. weights = (b - a) / 2 * weights
    for i in 1:N
        mul!(view(M, (i-1)*d+1:i*d, 1:D), sqrt(weights[i]), chol_integrand(nodes[i]))
    end
    Qh_R = qr!(M).R

    return Ah, Qh_R
end
