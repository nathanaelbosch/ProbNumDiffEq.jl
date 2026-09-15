"""
    SmootherState{T}

Per-step forward-pass quantities needed for the √MBF backward smoother: the innovation
`z` and the measurement Jacobian `H`. Both are independent of the diffusion/calibration
scale, so they can be stored during the forward pass and reused as-is during backward
smoothing (which runs after diffusion calibration has rescaled `x_filt`). The Kalman
gain and measurement-covariance Cholesky factor are instead recomputed fresh during
backward smoothing from the (possibly rescaled) filtered covariances, so they are always
consistent with the covariances they are paired with.
"""
struct SmootherState{T}
    z::Vector{T}
    H::Matrix{T}
end
function Base.copy(s::SmootherState)
    SmootherState(copy(s.z), copy(s.H))
end
