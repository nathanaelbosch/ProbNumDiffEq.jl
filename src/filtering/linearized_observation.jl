"""
    LinearizedObservation(m, z, H, R=nothing)

The observation ``y = h(x) + v``, ``v \\sim \\mathcal{N}(0, R)``, of the state ``x``,
linearized at the point `m`:
```math
h(x) - y ≈ z + H (x - m).
```
A first-order Taylor expansion gives `z = h(m) - y`, the negative of the innovation, and the
Jacobian `H = h'(m)`; the examples below make other choices. The data ``y`` enters only
through `z`. `m` has the dimension of the state, `z` that of the observation. `R` is the
noise covariance as a `PSDMatrix`, or `nothing` for an exact observation. As an affine
model ``y \\sim \\mathcal{N}(H x + b, R)``, it is `z = H m + b - y`.

[`update!`](@ref) evaluates the linearization at the mean ``μ`` of the state it conditions, as
``z + H (μ - m)``, which is `z` if `m` is ``μ``.

How `m`, `z` and `H` are obtained depends on the observation and on how it is linearized.
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
- A statistical linearization (not implemented yet), as in an unscented Kalman filter,
  linearizes over a distribution ``\\mathcal{N}(m, Σ)``: `z = E[h(x)] - y`,
  `H = Cov[h(x), x] Σ⁻¹`, and the covariance of its linearization error is added to `R`.

`H` and `R` are matrices of the same structured type as the state covariance (`Matrix`,
`IsometricKroneckerProduct` or `BlocksOfDiagonals`); with a `BlocksOfDiagonals` covariance,
`H` can also be a `PartialBlocksOfDiagonals`, which observes only some of the blocks.
Another type of `H` works with [`update!`](@ref) if the internal `_update_mean!` and
`_update_cov!` have methods for it, and `_matmul!` if `m` is not the mean, as for
[`linear_observation`](@ref), which also needs `size(H, 2)`. A matrix-free operator would add
these methods.
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

The [`LinearizedObservation`](@ref) of `y` in the linear model `y ~ N(H x, R)` of the state
`x`. The model is its own linearization at any point; at `m = 0`, with `m = Zeros(D)` and
`D = size(H, 2)`, it is `z = -y`.
"""
linear_observation(H, y, R=nothing) =
    LinearizedObservation(Zeros{eltype(y)}(size(H, 2)), -y, H, R)
