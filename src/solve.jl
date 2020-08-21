function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::AbstractODEFilter, args...; kwargs...)
    integrator = DiffEqBase.__init(prob, alg; args..., kwargs...)
    sol = DiffEqBase.solve!(integrator)
    return sol
end
DiffEqBase.__init(prob::DiffEqBase.AbstractODEProblem, alg::EKF0, args...; kwargs...) =
    DiffEqBase.__init(prob, ODEFilter(), args...; method=:ekf0, kwargs...)
DiffEqBase.__init(prob::DiffEqBase.AbstractODEProblem, alg::EKF1, args...; kwargs...) =
    DiffEqBase.__init(prob, ODEFilter(), args...; method=:ekf1, kwargs...)

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

                           precond_dt = dt == 0 ? 1e-2 : dt,

                           sigmarule=:schober,
                           local_errors=:schober,

                           progressbar=false,
                           maxiters=1e5,
                           saveat=nothing,
                           save_everystep=true,
                           internalnorm = DiffEqBase.ODE_DEFAULT_NORM,

                           timeseries_errors=false,
                           dense_errors=false,

                           kwargs...)
    # Init
    IIP = DiffEqBase.isinplace(prob)
    # @info "Called init" method steprule dt q

    if method == :ekf1 && isnothing(prob.f.jac)
        error("""EKF1 requires the Jacobian. To automatically generate it with ModelingToolkit.jl
               use ProbNumoDE.remake_prob_with_jac(prob).""")
    end

    if length(prob.u0) == 1 && size(prob.u0) == ()
        @warn "prob.u0 is a scalar; In order to run, we remake the problem with u0 = [u0]."
        prob = remake(prob, u0=[prob.u0])
    end

    (timeseries_errors != false) && @warn("`timeseries_errors` currently not supported")
    (dense_errors != false) && @warn("`dense_errors` currently not supported")


    f = prob.f
    u0 = prob.u0
    t0, tmax = prob.tspan
    p = prob.p
    d = length(u0)

    # Model
    constants = GaussianODEFilterConstantCache(prob, q, prior, method, precond_dt)

    # Cache
    cache = GaussianODEFilterCache(d, q, prob, constants, initialize_derivatives)

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
        :fixedWeightedMLE => WeightedMLESigma(),
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
    dt_init = dt != 0 ? dt : 1e-3
    opts = DEOptions(maxiters, adaptive, abstol, reltol, gamma, qmin, qmax, internalnorm, dtmin, dtmax)

    return ODEFilterIntegrator{IIP, typeof(u0), typeof(t0), typeof(p), typeof(f)}(
        f, u0, t0, t0, t0, tmax, dt_init, p, one(eltype(prob.tspan)),
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
