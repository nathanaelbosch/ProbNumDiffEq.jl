########################################################################################
# Structure following https://github.com/SciML/SimpleDiffEq.jl/tree/master/src/rk4
########################################################################################


########################################################################################
# Algorithm
########################################################################################
abstract type AbstractODEFilter <: DiffEqBase.AbstractODEAlgorithm end
mutable struct ODEFilter <: AbstractODEFilter end
mutable struct EKF0 <: AbstractODEFilter end
mutable struct EKF1 <: AbstractODEFilter end


########################################################################################
# Options
########################################################################################
mutable struct DEOptions
    maxiters
    adaptive
    abstol
    reltol
    gamma
    qmin
    qmax
    # beta1
    # beta2
    internalnorm
    dtmin
    dtmax
end




########################################################################################
# Integrator
########################################################################################
mutable struct ODEFilterIntegrator{IIP, S, T, P, F} <: DiffEqBase.AbstractODEIntegrator{ODEFilter, IIP, S, T}
    f::F                  # eom
    u::S                  # current functionvalue
    # x::X                  # current state
    t::T                  # current time
    t_new::T                  # next time
    t0::T                 # initial time, only for reinit
    tmax::T               # end of the interval
    dt::T                 # step size
    p::P                  # parameter container
    EEst                  # (Scaled) error estimate

    constants
    cache

    # Options
    opts::DEOptions       # General (not PN-specific) solver options
    sigma_estimator
    error_estimator
    steprule
    smooth::Bool          # Smooth the solution or not

    # Save into
    proposal              # Current proposal
    proposals             # List of proposals
    state_estimates       # List of state estimates, used to build the solution
    times

    # Misc
    iter::UInt            # Current iteration count
    accept_step::Bool     # If the current step is accepted
    retcode::Symbol       # Current return code, used to build the solution
    prob                  # Only used to build the solution
    alg                   # Only used to build the solution
    destats::DiffEqBase.DEStats   # To track stats like the number of f evals
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP




########################################################################################
# Caches
########################################################################################
abstract type ProbNumODECache <: DiffEqBase.DECache end
abstract type ProbNumODEConstantCache <: ProbNumODECache end
abstract type ProbNumODEMutableCache <: ProbNumODECache end
struct GaussianODEFilterConstantCache <: ProbNumODEMutableCache
    d                # Dimension of the problem
    q                # Order of the prior
    # dm                    # Dynamics Model
    # mm                    # Measurement Model
    A!
    Q!
    h!
    H!
    R
    # Precond
    # Precond_inv
    E0
    E1
    jac
    Precond
    InvPrecond
end
function GaussianODEFilterConstantCache(d, q, f, prior, method, precond_dt=0.5)
    @assert prior == :ibm
    Precond, InvPrecond = preconditioner(precond_dt, d, q)
    A!, Q! = ibm(d, q; precond_dt=precond_dt)

    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E0 = E0 * InvPrecond
    E1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E1 = E1 * InvPrecond

    @assert method in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    jac = nothing
    if method == :ekf1
        if !isnothing(f.jac)
            jac = f.jac
        else
            @warn """EKF1 requires the Jacobian, but it has not been explicitly passed to
                  the ODEProblem. We'll use ForwardDiff to compute it."""
            jac = (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u)
        end
    end
    h!(h, du, m) = h .= E1*m - du
    H!(H, ddu) = H .= E1 - ddu * E0
    R = zeros(d, d)

    return GaussianODEFilterConstantCache(d, q, A!, Q!, h!, H!, R, E0, E1, jac, Precond, InvPrecond)
end


mutable struct GaussianODEFilterCache <: ProbNumODEMutableCache
    u
    u_pred
    # tmp
    # utilde
    x
    x_pred
    x_filt
    measurement
    Ah
    Qh
    h
    H
    du
    ddu
    K
    σ_sq
    σ_sq_prev
    P_tmp
    err_tmp
end
function GaussianODEFilterCache(d, q, prob, constants, initialize_derivatives=true)
    @unpack Precond, InvPrecond, E0, E1 = constants
    u0 = prob.u0
    t0 = prob.tspan[1]
    # Initial states
    m0, P0 = initialize_derivatives ?
        initialize_with_derivatives(prob, q) :
        initialize_without_derivatives(prob, q)
    x0 = Precond * Gaussian(m0, P0)

    Ah_empty = diagm(0=>ones(d*(q+1)))
    Qh_empty = zeros(d*(q+1), d*(q+1))
    h = E1 * x0.μ
    H = zeros(d, d*(q+1))
    du = copy(h)
    ddu = zeros(d, d)
    sigma = 1.0
    v, S = copy(h), zeros(d,d)
    measurement = Gaussian(v, S)
    K = zeros(d*(q+1),d)

    GaussianODEFilterCache(
        copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0),
        measurement,
        Ah_empty, Qh_empty, h, H, du, ddu, K, sigma, sigma,
        copy(P0),
        copy(u0),
    )

end


function initialize_with_derivatives(prob, order)
    q = order
    d = length(prob.u0)
    m0 = isinplace(prob) ?
        _get_init_derivatives_mtk(prob, order) :
        get_initial_derivatives(prob, q)
    P0 = zeros(d*(q+1), d*(q+1))
    return m0, P0
end

function initialize_without_derivatives(prob, order, var=1e-3)
    q = order
    u0 = prob.u0
    d = length(u0)
    p = prob.p
    t0 = prob.tspan[1]

    m0 = zeros(d*(q+1))
    m0[1:d] = u0
    if !isinplace(prob)
        m0[d+1:2d] = prob.f(u0, p, t0)
    else
        prob.f(m0[d+1:2d], u0, p, t0)
    end
    P0 = [zeros(d, d) zeros(d, d*q);
          zeros(d*q, d) diagm(0 => var .* ones(d*q))]
    return m0, P0
end
