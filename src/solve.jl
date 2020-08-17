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
                            dt=0.1,
                            q=1,
                            saveat=nothing,
                            save_everystep=true,
                            abstol=1e-6, reltol=1e-3, ρ=0.95,
                            qmin=0.1, qmax=5.0,
                            method=:ekf1,
                            sigmarule=Schober16Sigma(),
                            steprule=:baseline,
                            progressbar=false,
                            maxiters=1e5,
                            smooth=true,
                            prior=:ibm,
                           initialize_derivatives=true,
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
    adaptive = steprule != :constant
    gamma = ρ
    steprules = Dict(
        :constant => constant_steprule(),
        :baseline => classic_steprule(abstol, reltol; ρ=ρ),
        :schober16 => schober16_steprule(;ρ=ρ, abstol=abstol, reltol=reltol),
    )
    steprule = steprules[steprule]

    empty_proposal = ()
    empty_proposals = []

    destats = DiffEqBase.DEStats(0)

    state_estimates = StructArray([cache.x])
    times = [t0]
    accept_step = false
    retcode = :Default

    opts = DEOptions(maxiters, adaptive, abstol, reltol, gamma, qmin, qmax)

    return ODEFilterIntegrator{IIP, typeof(u0), typeof(t0), typeof(p), typeof(f)}(
        f, u0, t0, t0, t0, tmax, dt, p,
        constants, cache,
        # d, q, dm, mm, sigmarule, steprule,
        opts, sigmarule, steprule, smooth,
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
