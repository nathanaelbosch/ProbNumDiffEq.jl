OrdinaryDiffEq.alg_autodiff(alg::AbstractEKF) = true
OrdinaryDiffEq.get_chunksize(alg::AbstractEKF) = Val(0)
OrdinaryDiffEq.isfsal(alg::AbstractEKF) = false


############################################
# Step size control
OrdinaryDiffEq.isadaptive(alg::AbstractEKF) = true
OrdinaryDiffEq.alg_order(alg::AbstractEKF) = alg.order+1
# OrdinaryDiffEq.alg_adaptive_order(alg::AbstractEKF) =

# PI control is the default!
OrdinaryDiffEq.isstandard(alg::AbstractEKF) = false # proportional
OrdinaryDiffEq.ispredictive(alg::AbstractEKF) = false # not sure, maybe Gustafsson acceleration?

# OrdinaryDiffEq.qmin_default(alg::AbstractEKF) =
# OrdinaryDiffEq.qmax_default(alg::AbstractEKF) =
OrdinaryDiffEq.beta2_default(alg::AbstractEKF) = 2//(5(alg.order+1))
OrdinaryDiffEq.beta1_default(alg::AbstractEKF, beta2) = 7//(10(alg.order+1))
# OrdinaryDiffEq.gamma_default(alg::AbstractEKF) =

# OrdinaryDiffEq.uses_uprev(alg::, adaptive::Bool) = adaptive
