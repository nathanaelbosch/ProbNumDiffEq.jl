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
                            kwargs...)
    # Init
    IIP = DiffEqBase.isinplace(prob)

    f = prob.f
    u0 = prob.u0
    t0, tmax = prob.tspan
    p = prob.p
    d = length(u0)

    # Model
    dm = ibm(q, d)
    mm = measurement_model(method, d, q, f, p, IIP)

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

    if eltype(m0) <: Measurement
        P0 = diagm(0 => Measurements.uncertainty.(m0) .^ 2)
        m0 = Measurements.value.(m0)
    else
        P0 = diagm(0 => [zeros(d); ones(d*q)] .+ 1e-16)
    end
    x0 = Gaussian(m0, P0)
    X = typeof(x0)

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

    state_estimates = StructArray([(t=t0, x=x0)])
    accept_step = false
    retcode = :Default

    opts = DEOptions(maxiters, adaptive, abstol, reltol, gamma, qmin, qmax)

    return ODEFilterIntegrator{IIP, typeof(u0), typeof(x0), typeof(t0), typeof(p), typeof(f)}(
        f, u0, _copy(x0), t0, t0, tmax, dt, p,
        d, q, dm, mm, sigmarule, steprule,
        empty_proposal, empty_proposals, 0,
        state_estimates, accept_step, retcode, prob, alg, smooth, destats, opts
    )
end


function DiffEqBase.solve!(integ::ODEFilterIntegrator)
    while integ.t < integ.tmax
        step!(integ)
    end
    postamble!(integ)
    sol = DiffEqBase.build_solution(
        integ.prob, integ.alg,
        integ.state_estimates.t,
        StructArray(integ.state_estimates.x),
        integ.proposals, integ;
        destats=integ.destats)
end
