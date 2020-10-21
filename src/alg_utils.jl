# qmin_default(alg::AbstractEKF) =
# qmax_default(alg::AbstractEKF) =
# alg_order(alg::AbstractEKF) =
# alg_adaptive_order(alg::AbstractEKF) =
beta2_default(alg::AbstractEKF) = 2//(5(alg.order+1))
beta1_default(alg::AbstractEKF, beta2) = 7//(10(alg.order+1))
# gamma_default(alg::AbstractEKF) =
# uses_uprev(alg::, adaptive::Bool) = adaptive
