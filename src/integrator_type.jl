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
mutable struct ODEFilterIntegrator{IIP, S, X, T, P, F} <: DiffEqBase.AbstractODEIntegrator{ODEFilter, IIP, S, T}
    f::F                  # eom
    u::S                  # current functionvalue
    x::X                  # current state
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    tmax::T               # end of the interval
    dt::T                 # step size
    p::P                  # parameter container

    # My additions
    d::Int                # Dimension of the problem
    q::Int                # Order of the prior
    dm                    # Dynamics Model
    mm                    # Measurement Model
    sigma_estimator
    steprule
    proposal              # Current proposal
    proposals             # List of proposals
    iter::UInt            # Current iteration count
    state_estimates       # List of state estimates, used to build the solution
    accept_step::Bool     # If the current step is accepted
    retcode::Symbol       # Current return code, used to build the solution
    prob                  # Only used to build the solution
    alg                   # Only used to build the solution
    smooth::Bool          # Smooth the solution or not
    destats::DiffEqBase.DEStats   # To track stats like the number of f evals
    opts::DEOptions       # Other general solver options, see above
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP
