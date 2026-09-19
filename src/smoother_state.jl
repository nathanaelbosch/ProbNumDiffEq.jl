"""
    SmootherState{T,TH,TK,TS,TR}

Per-step forward-pass quantities needed for the √MBF backward smoother: the measurement mean
`z = h(x_pred)` (the filter's innovation is `-z`, since the update assumes zero measurements),
the measurement Jacobian `H`, the Kalman gain `K`, and the Cholesky factor `S_U` of
the measurement covariance `S = H Σ_pred Hᵀ`. All four are the exact quantities that the
forward filter's update step used, so they can be stored during the forward pass and
reused as-is during backward smoothing. If the solution is recalibrated after the solve
(see [`calibrate_solution!`](@ref)), the stored `S_U` factors are rescaled accordingly,
while the gains `K = Σ_pred Hᵀ S⁻¹` are invariant under the rescaling.

In addition, `rate_parameter` stores the prior's rate parameter value as it was used at
this step: the transition matrices only depend on the step size for the IWP prior, but the
IOUP prior with `update_rate_parameter=true` updates its rate parameter (Rosenbrock-style)
at every step, so the backward pass has to restore the per-step value before recomputing
the transition matrices with that step's step size. It is `nothing` for priors whose
transitions only depend on the step size.

`H` and `K` keep whatever structured type the cache uses (dense `Matrix`,
`IsometricKroneckerProduct` for EK0, or `BlocksOfDiagonals` for DiagonalEK1) so that
backward smoothing can dispatch on them and exploit the same structure the forward pass
does, instead of forcing an expensive dense representation. `S_U` is stored in the same
structure as the measurement covariance itself.
"""
struct SmootherState{T,TH,TK,TS,TR}
    z::Vector{T}
    H::TH
    K::TK
    S_U::TS
    rate_parameter::TR
end
