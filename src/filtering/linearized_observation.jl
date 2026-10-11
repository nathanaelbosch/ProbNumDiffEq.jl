"""
    LinearizedObservation(m, z, H, R=nothing)

The observation ``y = h(x) + v``, ``v \\sim \\mathcal{N}(0, R)``, of the state ``x``,
linearized at the point `m`:
```math
h(x) - y ≈ z + H (x - m).
```
A first-order Taylor expansion gives `z = h(m) - y` and `H = h'(m)`, but the examples below
show other choices. For an affine model ``y \\sim \\mathcal{N}(H x + b, R)``, it is
`z = H m + b - y`. `R` is the noise covariance as a `PSDMatrix`, or `nothing` for an exact
observation.

[`update!`](@ref) evaluates the linearization at the mean ``μ`` of the state it conditions,
as ``z + H (μ - m)``, which is `z` if `m` is ``μ``.

Examples:
- The ODE step observes ``y = 0`` through the residual ``h(x) = E_1 x - f(E_0 x, t)`` of the
  ODE ``u' = f(u, t)``, linearized at the predicted mean ``μ``: `m = μ`, `z = h(μ)` and
  `H = E_1 - J E_0`, where `J` is the Jacobian of `f` in the `EK1`, its diagonal in the
  `DiagonalEK1`, and `0` in the `EK0`.
- Data, as in `DataUpdateCallback` and Fenrir, is linear, ``y = C E_0 x + v``, so the
  linearization is exact at any `m`. [`linear_observation`](@ref) takes `m = 0`, which gives
  `z = -y` and `H = C E_0`.
- An iterated extended Kalman filter, as in `ManifoldUpdate`, conditions the same prediction
  ``\\mathcal{N}(μ, Σ)`` in every iteration and linearizes ``h`` at the conditioned mean of
  the previous iteration, so `m = μ` only in the first.

`H` and `R` have the same structure as the state covariance: dense,
`IsometricKroneckerProduct` or `BlocksOfDiagonals`. With a `BlocksOfDiagonals` covariance,
`H` can also be a `BlocksOfScaledSelections`, which observes only some of the blocks. Other
types of `H` need methods for the internal `_update_mean!` and `_update_cov!`, and for
`_matmul!` if `m` is not the mean.
"""
struct LinearizedObservation{mT,zT,HT,RT}
    m::mT
    z::zT
    H::HT
    R::RT
end
LinearizedObservation(m, z, H) = LinearizedObservation(m, z, H, nothing)

"""
    linear_observation(H, y, R=nothing)

The [`LinearizedObservation`](@ref) of `y` in the linear model
``y \\sim \\mathcal{N}(H x, R)``, linearized at `m = 0`, which is exact and gives `z = -y`.
"""
linear_observation(H, y, R=nothing) =
    LinearizedObservation(Zeros{eltype(y)}(size(H, 2)), -y, H, R)
