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
# Integrator
########################################################################################
mutable struct ODEFilterIntegrator{IIP, S, X, T, P, F} <: DiffEqBase.AbstractODEIntegrator{ODEFilter, IIP, S, T}
    f::F                  # eom
    u::S                  # current functionvalue
    x::X                  # current state
    xprev::X              # previous state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    x_modified::Bool
    # ks::Vector{S}         # interpolants of the algorithm
    # cs::SVector{6, T}     # ci factors cache: time coefficients
    # as::SVector{21, T}    # aij factors cache: solution coefficients
    # rs::SVector{22, T}    # rij factors cache: interpolation coefficients

    # My additions
    d::Int
    q::Int
    dm
    mm
    sigma_estimator
    steprule
    proposal
    proposals
    iter::UInt
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP


########################################################################################
# Initialization
########################################################################################
function odefilter_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P, q::Integer, method,
                        sigmarule, steprule, abstol, reltol, ρ, prob_kwargs) where {F, P, T, S}
    d = length(u0)
    dm = ibm(q, d)
    mm = measurement_model(method, d, q, f, p, IIP)

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

    steprules = Dict(
        :constant => constant_steprule(),
        :pvalue => pvalue_steprule(0.05),
        :baseline => classic_steprule(abstol, reltol; ρ=ρ),
        :measurement_error => measurement_error_steprule(;abstol=abstol, reltol=reltol, ρ=ρ),
        :measurement_scaling => measurement_scaling_steprule(),
        :schober16 => schober16_steprule(;ρ=ρ, abstol=abstol, reltol=reltol),
    )
    steprule = steprules[steprule]

    empty_proposal = ()
    empty_proposals = []

    return ODEFilterIntegrator{IIP, S, X, T, P, F}(
        f, u0, _copy(x0), _copy(x0), t0, t0, t0, dt, sign(dt), p, true,
        d, q, dm, mm, sigmarule, steprule, empty_proposal, empty_proposals, 0
    )
end


########################################################################################
# Solve
########################################################################################
function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::ODEFilter;
                            dt=0.1,
                            q=1,
                            saveat=nothing,
                            save_everystep=true,
                            abstol=1e-6, reltol=1e-3, ρ=0.95,
                            method=:ekf1,
                            sigmarule=Schober16Sigma(),
                            steprule=:baseline,
                            progressbar=false,
                            maxiters=1e5,
                            smoothed=true,
                            kwargs...)
    # Init
    IIP = DiffEqBase.isinplace(prob)
    integ = odefilter_init(prob.f, IIP, prob.u0, prob.tspan[1], dt, prob.p, q, method, sigmarule, steprule, abstol, reltol, ρ, prob.kwargs)
    # More Initialization
    t_0, T = prob.tspan
    sol = StructArray([(t=integ.t, x=integ.x)])
    proposals = []
    retcode = :Success

    # Solve
    if progressbar pbar_update, pbar_close = make_progressbar(0.1) end
    iter = 0
    while integ.t < T
        if progressbar pbar_update(fraction=(integ.t-t_0)/(T-t_0)) end

        step!(integ)
        push!(sol, (t=integ.t, x=integ.x))

        if integ.iter >= maxiters
            retcode = :MaxIters
            break
        end
    end
    if progressbar pbar_close() end

    smoothed && smooth!(sol, integ)
    calibrate!(sol, integ)

    # Format Solution
    sol = DiffEqBase.build_solution(prob, alg, sol.t, StructArray(sol.x),
                                    proposals, integ, retcode=retcode)

    return sol
end
DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::EKF0; kwargs...) =
    DiffEqBase.__solve(prob, ODEFilter(); method=:ekf0, kwargs...)
DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::EKF1; kwargs...) =
    DiffEqBase.__solve(prob, ODEFilter(); method=:ekf1, kwargs...)



########################################################################################
# Step
########################################################################################
function DiffEqBase.step!(integ::ODEFilterIntegrator{IIP, S, X, T}) where {IIP, S, X, T}
    accept = false
    while !accept
        integ.iter += 1
        t = integ.t + integ.dt
        prediction, A, Q = predict(integ)
        h, H = measure(integ, prediction, t)

        σ_sq = dynamic_sigma_estimation(
            integ.sigma_estimator; integ, prediction=prediction, v=h, H=H, Q=Q)
        prediction = Gaussian(prediction.μ, prediction.Σ + (σ_sq - 1) * Q)

        filter_estimate, measurement = update(integ, prediction, h, H)

        proposal = (t=t,
                prediction=prediction,
                filter_estimate=filter_estimate,
                measurement=measurement,
                H=H, Q=Q, v=h,
                σ²=σ_sq)

        integ.proposal = proposal

        accept, dt_proposal = integ.steprule(integ)
        push!(integ.proposals, (proposal..., accept=accept, dt=integ.dt))
        integ.dt = dt_proposal

        if accept
            integ.x = proposal.filter_estimate
            integ.t = proposal.t
            # integ.dt = min(integ.dt, T-integ.t)
        end
    end
end


function predict(integ)
    @unpack dm, mm, x, t, dt = integ

    m, P = x.μ, x.Σ
    A, Q = dm.A(dt), dm.Q(dt)
    m_p = A * m
    P_p = Symmetric(A*P*A') + Q
    prediction=Gaussian(m_p, P_p)
    return prediction, A, Q
end


function measure(integ, prediction, t)
    @unpack mm = integ
    m_p = prediction.μ
    h = mm.h(m_p, t)
    H = mm.H(m_p, t)
    return h, H
end


function update(integ, prediction, h, H)
    R = integ.mm.R
    v = 0 .- h

    m_p, P_p = prediction.μ, prediction.Σ
    S = Symmetric(H * P_p * H' + R)
    K = P_p * H' * inv(S)
    m = m_p + K*v
    P = P_p - Symmetric(K*S*K')
    return Gaussian(m, P), Gaussian(v, S)
end
