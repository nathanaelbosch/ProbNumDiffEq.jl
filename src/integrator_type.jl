########################################################################################
# Structure following https://github.com/SciML/SimpleDiffEq.jl/tree/master/src/rk4
########################################################################################


########################################################################################
# Options
########################################################################################
mutable struct DEOptions{MI, absType, relType, QT, F1, tType, F2}
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
    unstable_check::F2
    dtmin::tType
    dtmax::tType
    force_dtmin::Bool
    verbose::Bool
end


########################################################################################
# Integrator
########################################################################################
mutable struct ODEFilterIntegrator{
    IIP, S, T, P, F, QT, O, cacheType, xType, algType,
} <: DiffEqBase.AbstractODEIntegrator{algType, IIP, S, T}
    sol
    f::F                               # eom
    u::S                               # current functionvalue
    # x::X                             # current state
    t::T                               # current time
    dt::T                              # step size
    p::P                               # parameter container
    EEst::QT                           # (Scaled) error estimate
    qold::QT                           #

    cache::cacheType

    # Options
    opts::O                            # General (not PN-specific) solver options

    # Misc
    iter::UInt                         # Current iteration count
    success_iter::UInt                 # Number of successful steps
    accept_step::Bool                  # If the current step is accepted
    retcode::Symbol                    # Current return code, used to build the solution
    alg::algType                       # Only used to build the solution
    destats::DiffEqBase.DEStats        # To track stats like the number of f evals
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP
