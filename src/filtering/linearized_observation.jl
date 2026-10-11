"""
    LinearizedObservation(m, r, H, R=nothing)

Linear(ized) observation of a state ``x``, of the form ``\\mathcal{N}(H (x - m) + r, R)``,
observed to be zero. This typically results from linearizing a nonlinear observation
model ``h(x)`` at a point `m`, with `r = h(m)` and `H = J_h(m)` the Jacobian of `h` at `m`,
as is the standard case in ODE filtering.

Examples:
- The ODE filter's nonlinear observation model ``h(x) = E_1 x - f(E_0 x, t)`` is at each
  step linearized at the predicted mean ``μ``, leading to `m = μ`, `r = h(μ)` and
  `H = E_1 - J E_0`, where `J` is the Jacobian of `f` in the `EK1` (or its diagonal in the
  `DiagonalEK1`, or `0` in the `EK0`).
- Non-zero observations, such as external observations of the ODE trajectory, can also be
  represented in this model by shifting the observation model by the observation `y`.
  Since a linear observation ``y \\sim \\mathcal{N}(M E_0 x, R)`` is equivalent to
  ``0 \\sim \\mathcal{N}(M E_0 x - y, R)``, this gives `H = M E_0`, `m = 0` as we do not
  have to linearize, and `r = -y`.
- An iterated extended Kalman filter, as in `ManifoldUpdate`, conditions the same prediction
  ``\\mathcal{N}(μ, Σ)`` in every iteration and linearizes ``h`` at the conditioned mean of
  the previous iteration, so `m = μ` only in the first.

`H` and `R` have the same structure as the state covariance: dense,
`IsometricKroneckerProduct` or `BlocksOfDiagonals`. With a `BlocksOfDiagonals` covariance,
`H` can also be a `BlocksOfScaledSelections`, which observes only some of the blocks. Other
types of `H` need methods for the internal `_update_mean!` and `_update_cov!`, and for
`_matmul!` if `m` is not the mean.
"""
struct LinearizedObservation{mT,rT,HT,RT}
    m::mT
    r::rT
    H::HT
    R::RT
end
LinearizedObservation(m, r, H) = LinearizedObservation(m, r, H, nothing)

"""
    linear_observation(H, y, R=nothing)

Builds the [`LinearizedObservation`](@ref) corresponding to the linear model
``y \\sim \\mathcal{N}(H x, R)`` with observation `y`.
"""
linear_observation(H, y, R=nothing) =
    LinearizedObservation(Zeros{eltype(y)}(size(H, 2)), -y, H, R)
