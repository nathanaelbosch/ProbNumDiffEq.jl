function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::AbstractODEFilter, args...; kwargs...)
  integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
  sol = DiffEqBase.solve!(integrator)
  return sol
end
DiffEqBase.__init(prob::DiffEqBase.AbstractODEProblem, alg::EKF0; kwargs...) =
    DiffEqBase.__init(prob, ODEFilter(); method=:ekf0, kwargs...)
DiffEqBase.__init(prob::DiffEqBase.AbstractODEProblem, alg::EKF1; kwargs...) =
    DiffEqBase.__init(prob, ODEFilter(); method=:ekf1, kwargs...)

function DiffEqBase.__init(prob::DiffEqBase.AbstractODEProblem, alg::ODEFilter;

                           method=:ekf1,
                           prior=:ibm,
                           q=1,
                           smooth=true,
                           initialize_derivatives=true,

                           steprule=:standard,
                           dt=eltype(prob.tspan)(0),
                           abstol=1e-6, reltol=1e-3, gamma=9//10,
                           qmin=0.1, qmax=5.0,
                           dtmin=nothing,
                           dtmax=eltype(prob.tspan)((prob.tspan[end]-prob.tspan[1])),

                           sigmarule=:schober,
                           local_errors=:schober,

                           progressbar=false,
                           maxiters=1e5,
                           saveat=nothing,
                           save_everystep=true,
                           internalnorm = DiffEqBase.ODE_DEFAULT_NORM,
                           kwargs...)
    # Init
    IIP = DiffEqBase.isinplace(prob)

    f = prob.f
    u0 = prob.u0
    t0, tmax = prob.tspan
    p = prob.p
    d = length(u0)

    # Model
    constants = GaussianODEFilterConstantCache(d, q, f, prior, method)

    # Cache
    cache = GaussianODEFilterCache(d, q, prob, initialize_derivatives)

    # Solver Options
    tType = eltype(prob.tspan)
    adaptive = steprule != :constant
    if !adaptive && dt == tType(0)
        error("Fixed timestep methods require a choice of dt")
    end
    steprules = Dict(
        :constant => ConstantSteps(),
        :standard => StandardSteps(),
        :pi => PISteps(),
    )
    steprule = steprules[steprule]

    error_estimators = Dict(
        :schober => SchoberErrors(),
        :prediction => PredictionErrors(),
        :filtering => FilterErrors(),
    )
    error_estimator = error_estimators[local_errors]

    sigmarules = Dict(
        :schober => SchoberSigma(),
        :fixedMLE => MLESigma(),
        :fixedMAP => MAPSigma(),
    )
    sigmarule = sigmarules[sigmarule]

    empty_proposal = ()
    empty_proposals = []

    destats = DiffEqBase.DEStats(0)

    state_estimates = StructArray([cache.x])
    times = [t0]
    accept_step = false
    retcode = :Default

    isnothing(dtmin) && (dtmin = DiffEqBase.prob2dtmin(prob; use_end_time=true))
    opts = DEOptions(maxiters, adaptive, abstol, reltol, gamma, qmin, qmax, internalnorm, dtmin, dtmax)

    return ODEFilterIntegrator{IIP, typeof(u0), typeof(t0), typeof(p), typeof(f)}(
        f, u0, t0, t0, t0, tmax, dt, p, one(eltype(prob.tspan)),
        constants, cache,
        # d, q, dm, mm, sigmarule, steprule,
        opts, sigmarule, error_estimator, steprule, smooth,
        #
        empty_proposal, empty_proposals, state_estimates, times,
        #
        0, accept_step, retcode, prob, alg, destats,
    )
end


function DiffEqBase.solve!(integ::ODEFilterIntegrator)
    while integ.t < integ.tmax
        step!(integ)
    end
    postamble!(integ)
    sol = DiffEqBase.build_solution(
        integ.prob, integ.alg,
        integ.times,
        integ.state_estimates,
        integ.proposals, integ;
        destats=integ.destats)
end
