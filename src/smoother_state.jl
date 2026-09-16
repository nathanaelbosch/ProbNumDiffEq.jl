"""
    SmootherState{T,TH}

Per-step forward-pass quantities needed for the √MBF backward smoother: the innovation
`z` and the measurement Jacobian `H`. Both are independent of the diffusion/calibration
scale, so they can be stored during the forward pass and reused as-is during backward
smoothing (which runs after diffusion calibration has rescaled `x_filt`). The Kalman
gain and measurement-covariance Cholesky factor are instead recomputed fresh during
backward smoothing from the (possibly rescaled) filtered covariances, so they are always
consistent with the covariances they are paired with.

`H` keeps whatever structured type `cache.H` naturally has (dense `Matrix`,
`IsometricKroneckerProduct` for EK0, or `BlocksOfDiagonals` for `DiagonalEK1`) so that
backward smoothing can dispatch on it and exploit the same structure the forward pass
does, instead of forcing an expensive dense `D×D` representation.
"""
struct SmootherState{T,TH<:AbstractMatrix{T}}
    z::Vector{T}
    H::TH
end
function Base.copy(s::SmootherState)
    SmootherState(copy(s.z), copy(s.H))
end
