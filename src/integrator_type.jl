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
mutable struct DEOptions{MI, absType, relType, QT, F1, tType}
    maxiters::MI
    adaptive::Bool
    abstol::absType
    reltol::relType
    gamma::QT
    qmin::QT
    qmax::QT
    beta1::QT
    beta2::QT
    qoldinit::QT
    internalnorm::F1
    dtmin::tType
    dtmax::tType
end




########################################################################################
# Integrator
########################################################################################
mutable struct ODEFilterIntegrator{
    IIP, S, T, P, F, QT, O, constantsType, cacheType,
    sigmaestType, errorestType, stepruleType, proposalsType,
    xType, sigmaType, probType, algType} <: DiffEqBase.AbstractODEIntegrator{algType, IIP, S, T}
    f::F                               # eom
    u::S                               # current functionvalue
    # x::X                             # current state
    t::T                               # current time
    t_new::T                           # next time
    t0::T                              # initial time, only for reinit
    tmax::T                            # end of the interval
    dt::T                              # step size
    p::P                               # parameter container
    EEst::QT                           # (Scaled) error estimate
    qold::QT                           #

    constants::constantsType
    cache::cacheType

    # Options
    opts::O                            # General (not PN-specific) solver options
    sigma_estimator::sigmaestType
    error_estimator::errorestType
    steprule::stepruleType
    smooth::Bool                       # Smooth the solution or not

    # Save into
    proposals::proposalsType           # List of proposals
    state_estimates::Vector{xType}     # List of state estimates, used to build the solution
    times::Vector{T}
    sigmas::Vector{sigmaType}

    # Misc
    iter::UInt                         # Current iteration count
    success_iter::UInt                 # Number of successful steps
    accept_step::Bool                  # If the current step is accepted
    retcode::Symbol                    # Current return code, used to build the solution
    prob::probType                     # Only used to build the solution
    alg::algType                       # Only used to build the solution
    destats::DiffEqBase.DEStats        # To track stats like the number of f evals
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP




########################################################################################
# Caches
########################################################################################
abstract type ProbNumODECache <: DiffEqBase.DECache end
abstract type ProbNumODEConstantCache <: ProbNumODECache end
abstract type ProbNumODEMutableCache <: ProbNumODECache end
struct GaussianODEFilterConstantCache{RType, EType, F1, F2} <: ProbNumODEMutableCache
    d::Int                  # Dimension of the problem
    q::Int                  # Order of the prior
    # dm                    # Dynamics Model
    # mm                    # Measurement Model
    A!
    Q!
    h!
    H!
    R::RType
    # Precond
    # Precond_inv
    E0::EType
    E1::EType
    jac
    Precond::F1
    InvPrecond::F2
end
function GaussianODEFilterConstantCache(prob, q, prior, method)
    d, f = length(prob.u0), prob.f
    @assert prior == :ibm
    Precond, InvPrecond = preconditioner(d, q)
    A!, Q! = ibm(d, q)

    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    # E0 = E0 * InvPrecond
    E1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    # E1 = E1 * InvPrecond

    @assert method in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    jac = method == :ekf1 ? f.jac : nothing
    h!(h, du, m) = h .= E1*m - du
    H!(H, ddu) = H .= E1 - ddu * E0
    # R = diagm(0 => 1e-10 .* ones(d))
    R = zeros(d, d)

    return GaussianODEFilterConstantCache{typeof(R), typeof(E0), typeof(Precond), typeof(InvPrecond)}(
        d, q, A!, Q!, h!, H!, R, E0, E1, jac, Precond, InvPrecond)
end


mutable struct GaussianODEFilterCache{uType, xType, matType, sigmaType} <: ProbNumODEMutableCache
    u::uType
    u_pred::uType
    u_filt::uType
    u_tmp::uType
    # tmp
    x::xType
    x_pred::xType
    x_filt::xType
    x_tmp::xType
    measurement
    Ah::matType
    Qh::matType
    h::uType
    H::matType
    du::uType
    ddu::matType
    K::matType
    σ_sq::sigmaType
    σ_sq_prev::sigmaType
    P_tmp::matType
    err_tmp::uType
end
function GaussianODEFilterCache(d, q, prob, constants, σ0, initialize_derivatives=true)
    @unpack E0, E1 = constants
    u0 = prob.u0

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    t0 = prob.tspan[1]
    # Initial states
    m0, P0 = initialize_derivatives ?
        get_initial_states_forwarddiff(prob, q) :
        initialize_without_derivatives(prob, q)
    # P0 += eps(eltype(P0)) * I
    x0 = Gaussian(m0, P0)

    Ah_empty = diagm(0=>ones(uElType, d*(q+1)))
    Qh_empty = zeros(uElType, d*(q+1), d*(q+1))
    h = E1 * x0.μ
    H = uElType.(zeros(d, d*(q+1)))
    du = copy(h)
    ddu = uElType.(zeros(d, d))
    sigma = σ0
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = copy(H')

    GaussianODEFilterCache{uType, typeof(x0), matType, typeof(sigma)}(
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        Ah_empty, Qh_empty, h, H, du, ddu, K, sigma, sigma,
        copy(P0),
        copy(u0),
    )

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
