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
