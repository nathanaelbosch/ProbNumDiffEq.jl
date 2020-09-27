function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::AbstractODEFilter, args...; kwargs...)
    @debug "Called solve with" args kwargs
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
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
                           smooth=false,
                           initialize_derivatives=true,

                           bigfloat=false,

                           steprule=:standard,
                           dt=eltype(prob.tspan)(0),
                           abstol=1e-6, reltol=1e-3,
                           # DiffEq defaults: Better for STIFF problems
                           gamma=9//10,
                           qmin=2//10, qmax=10,
                           # Schober Defaults
                           # gamma=0.95,
                           # qmin=0.1, qmax=5.0,
                           dtmin=DiffEqBase.prob2dtmin(prob; use_end_time=true),
                           dtmax=eltype(prob.tspan)((prob.tspan[end]-prob.tspan[1])),
                           # Some random manual tweaking lead to this
                           # beta1 = 1.2/(q+1),
                           # beta2 = 0.2/(q+1),
                           # OrdinaryDiffEq defaults: Works better for STIFF problems
                           beta1 = 7//(10(q+1)),
                           beta2 = 2//(5(q+1)),
                           # Bras paper betas (not 100% sure since they have different notation)
                           # beta1 = 0.07/(q+1),
                           # beta2 = 1.2/(q+1),

                           qoldinit = 1//10^4,

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
    bigfloat && (prob = remake(prob, u0=big.(prob.u0)))
    IIP = DiffEqBase.isinplace(prob)

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
    u0 = copy(prob.u0)
    t0, tmax = prob.tspan
    p = prob.p
    d = length(u0)

    # Model
    constants = GaussianODEFilterConstantCache(prob, q, prior, method)

    # Solver Options
    tType = eltype(prob.tspan)
    adaptive = steprule != :constant
    if !adaptive && dt == tType(0)
        error("Fixed timestep methods require a choice of dt")
    end
    steprules = Dict(
        :constant => ConstantSteps(),
        :standard => StandardSteps(),
        :PI => PISteps(),
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
        :schoberMV => MVSchoberSigma(),
        :EM => EMSigma(),
        :optim => OptimSigma(),
        :fixedMLE => MLESigma(),
        :MVfixedMLE => MVMLESigma(),
        :fixedMAP => MAPSigma(),
        :fixedWeightedMLE => WeightedMLESigma(),
    )
    sigmarule = sigmarules[sigmarule]

    # Cache
    cache = GaussianODEFilterCache(d, q, prob, constants, initial_sigma(sigmarule, d, q), initialize_derivatives)

    xType = typeof(cache.x)
    sigmaType = typeof(cache.σ_sq)
    measType = typeof(cache.measurement)
    proposalType = NamedTuple{(:t, :prediction, :filter_estimate, :measurement, :H, :Q, :v, :σ², :accept, :dt),
                              Tuple{tType, xType, xType, measType, typeof(cache.H), typeof(cache.Qh),
                                    typeof(cache.h), sigmaType, Bool, tType}}
    empty_proposals = proposalType[]

    destats = DiffEqBase.DEStats(0)

    state_estimates = StructArray([copy(cache.x)])
    times = [t0]
    sigmas = []
    accept_step = false
    retcode = :Default

    isnothing(dtmin) && (dtmin = DiffEqBase.prob2dtmin(prob; use_end_time=true))
    dt_init = dt != 0 ? dt : 1e-3
    QT = tType
    opts = DEOptions{typeof(maxiters), typeof(abstol), typeof(reltol), QT, typeof(internalnorm), tType}(
        maxiters, adaptive, abstol, reltol, QT(gamma), QT(qmin), QT(qmax),
        QT(beta1), QT(beta2), QT(qoldinit),
        internalnorm, dtmin, dtmax)

    return ODEFilterIntegrator{IIP, typeof(u0), typeof(t0), typeof(p), typeof(f), QT, typeof(opts), typeof(constants), typeof(cache),
                               typeof(sigmarule), typeof(error_estimator), typeof(steprule), typeof(empty_proposals),
                               xType, sigmaType, typeof(prob), typeof(alg)}(
        nothing, f, u0, t0, t0, t0, tmax, dt_init, p, one(QT), QT(qoldinit),
        constants, cache,
        # d, q, dm, mm, sigmarule, steprule,
        opts, sigmarule, error_estimator, steprule, smooth,
        #
        empty_proposals, state_estimates, times, sigmas,
        #
        0, 0, accept_step, retcode, prob, alg, destats,
    )
end


function DiffEqBase.solve!(integ::ODEFilterIntegrator)
    while integ.t < integ.tmax && integ.iter < integ.opts.maxiters
        step!(integ)
    end
    retcode = integ.iter == integ.opts.maxiters ? :MaxIters : :Success
    postamble!(integ)
    sol = DiffEqBase.build_solution(
        integ.prob, integ.alg,
        integ.times,
        integ.state_estimates,
        integ.sigmas,
        integ.proposals, integ;
        retcode=retcode,
        destats=integ.destats)
end
