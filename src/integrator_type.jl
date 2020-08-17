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

    constants
    cache

    # Options
    opts::DEOptions       # General (not PN-specific) solver options
    sigma_estimator
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
end
function GaussianODEFilterConstantCache(d, q, f, prior, method)
    @assert prior == :ibm
    A!, Q! = ibm(d, q)

    E0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    E1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))

    @assert method in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    jac = method == :ekf0 ? (args...) -> 0. :
        ((hasfield(typeof(f), :jac) && !isnothing(f.jac)) ? f.jac :
         (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u))
    h!(h, du, m) = h .= E1*m - du
    H!(H, ddu) = H .= E1 - ddu * E0
    R = zeros(d, d)

    return GaussianODEFilterConstantCache(d, q, A!, Q!, h!, H!, R, E0, E1, jac)
end


mutable struct GaussianODEFilterCache <: ProbNumODEMutableCache
    u
    uprev
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
end
function GaussianODEFilterCache(d, q, f, p, u0, t0, IIP)
    # Initial states
    initialize_derivatives = false
    initialize_derivatives = initialize_derivatives == :auto ? q <= 3 : false
    m0 = zeros(d*(q+1))
    if initialize_derivatives
        derivatives = get_derivatives((x, t) -> f(x, p, t), d, q)
        m0 = vcat(u0, [_f(u0, t0) for _f in derivatives]...)
    else
        m0[1:d] = u0
        if !IIP
            m0[d+1:2d] = f(u0, p, t0)
        else
            f(m0[d+1:2d], u0, p, t0)
        end
    end

    # if eltype(m0) <: Measurement
    #     P0 = diagm(0 => Measurements.uncertainty.(m0) .^ 2)
    #     m0 = Measurements.value.(m0)
    # else
    #     P0 = diagm(0 => [zeros(d); ones(d*q)] .+ 1e-16)
    # end
    P0 = zeros(d*(q+1), d*(q+1))
    x0 = Gaussian(m0, P0)
    # X = typeof(x0)

    Ah = diagm(0=>ones(d*(q+1)))
    Qh = zeros(d*(q+1), d*(q+1))
    h = zeros(d)
    H = zeros(d, d*(q+1))
    du = zeros(d)
    ddu = zeros(d, d)
    sigma = 1.0
    v, S = zeros(d), zeros(d,d)
    measurement = Gaussian(v, S)
    K = zeros(d*(q+1),d)

    GaussianODEFilterCache(
        copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0),
        measurement,
        Ah, Qh, h, H, du, ddu, K, sigma, sigma
    )

end
