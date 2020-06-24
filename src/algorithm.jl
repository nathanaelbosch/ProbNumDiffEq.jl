using LinearAlgebra
using Measurements
using Distributions
using ForwardDiff
using StructArrays
using DiffEqBase
using ProgressLogging
using UnPack


# Everything is a Gaussian here
mutable struct Gaussian{T<:AbstractFloat}
    μ::AbstractVector{T}
    Σ::AbstractMatrix{T}
    Gaussian{T}(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:AbstractFloat} =
        (length(μ) == size(Σ,1)) ?
        (Σ≈Σ' ? new(μ, Symmetric(Σ)) : error("Σ is not symmetric: $Σ")) :
        # (Σ≈Σ' ? new(μ, Σ) : error("Σ is not symmetric: $Σ")) :
        error("Wrong input dimensions: size(μ)=$(size(μ)), size(Σ)=$(size(Σ))")
end
Gaussian(μ::AbstractVector{T}, Σ::AbstractMatrix{T}) where {T<:AbstractFloat} =
    Gaussian{T}(μ, Σ)


struct StateBelief
    t::Real
    x::Gaussian
end



# Kalman Filter:
"""
No strict separation between prediction and update, since estimating σ at each step
requires a different order of comutation of the update and predict steps.
"""
function predict_update(solver, cache)
    @unpack dm, mm = solver
    @unpack t, x, dt = cache
    t = t + dt

    m, P = x.μ, x.Σ
    A, Q = dm.A(dt), dm.Q(dt)

    m_p = A * m

    H, R = mm.H(m_p, t), mm.R
    v = 0 .- mm.h(m_p, t)

    # Decide if constant sigma or variable sigma
    # σ² = solver.sigma_type == :fixed ? 1 :
    #     solver.sigma_estimator(;H=H, Q=Q, v=v)
    σ² = dynamic_sigma_estimation(solver.sigma_estimator; H=H, Q=Q, v=v)

    # @show P
    P_p = A*P*A' + σ²*Q
    S = H * P_p * H' + R
    # @show P_p
    # @show S
    K = P_p * H' * inv(S)
    m = m_p + K*v
    P = P_p - K*S*K'

    return (t=t,
            prediction=Gaussian(m_p, P_p),
            filter_estimate=Gaussian(m, P),
            measurement=Gaussian(v, S),
            σ²=σ²)
end


function smooth(filter_estimate::Gaussian,
                prediction::Gaussian,
                smoothed_estimate::Gaussian,
                dynamics_model)
    m, P = kf_smooth(
        filter_estimate.μ, filter_estimate.Σ,
        prediction.μ, prediction.Σ,
        smoothed_estimate.μ, smoothed_estimate.Σ,
        dynamics_model.A, dynamics_model.Q
    )
    return Gaussian(m, P)
end

function smooth(sol, solver, proposals)
    smoothed_solution = StructArray{StateBelief}(undef, length(sol))
    smoothed_solution[end] = sol[end]
    smoothed_solution[1] = sol[1]
    accepted_proposals = [p for p in proposals if p.accept]
    @show length(accepted_proposals), length(sol)
    for i in length(smoothed_solution)-1:-1:2
        h = accepted_proposals[i].dt  # step t -> t+1
        h2 = sol[i+1].t - sol[i].t
        @assert h ≈ h2

        prediction = accepted_proposals[i].prediction  # t+1
        filter_estimate = sol[i].x  # t
        smoothed_estimate = smoothed_solution[i+1].x # t+1

        smoothed_solution[i] = StateBelief(
            sol[i].t,
            smooth(filter_estimate,
                   prediction,
                   smoothed_estimate,
                   (A=solver.dm.A(h), Q=solver.dm.Q(h)))
        )
    end
    return smoothed_solution
end


# IBM
"""Generate the discrete dynamics for a q-IBM model

Careful: Dimensions are ordered differently than in `probnum`!"""
function ibm(q::Int, d::Int; σ::Int=1)
    F̃ = diagm(1 => ones(q))
    I_d = diagm(0 => ones(d))
    F = kron(F̃, I_d)  # In probnum the order is inverted

    # L̃ = zeros(q+1)
    # L̃[end] = σ^2
    # I_d = diagm(0 => ones(d))
    # L = kron(L̃, I_d)'  # In probnum the order is inverted

    A(h) = exp(F*(h))

    function Q(h)
        function _transdiff_ibm_element(row, col)
            idx = 2 * q + 1 - row - col
            fact_rw = factorial(q - row)
            fact_cl = factorial(q - col)

            return σ ^ 2 * (h ^ idx) / (idx * fact_rw * fact_cl)
        end

        qh_1d = [_transdiff_ibm_element(row, col) for col in 0:q, row in 0:q]
        I_d = diagm(0 => ones(d))
        return kron(qh_1d, I_d)
    end

    return (A=A, Q=Q, q=q, d=d)
end


# Measurement Model
function ekf0_measurement_model(d, q, ivp)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h(m, t) = H_1*m - ivp.f(H_0*m, ivp.p, t)
    H(m, t) = H_1
    return (h=h, H=H, R=R)
end


# Measurement Model
function ekf1_measurement_model(d, q, ivp)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    Jf(x, p, t) = (:jac in keys(ivp.kwargs)) ?
        ivp.kwargs[:jac](x, ivp.p, t) :
        ForwardDiff.jacobian(_u -> ivp.f(_u, ivp.p, t), x)
    h(m, t) = H_1*m - ivp.f(H_0*m, ivp.p, t)
    H(m, t) = H_1 - Jf(H_0*m, ivp.p, t) * H_0
    return (h=h, H=H, R=R)
end


Base.@kwdef struct Solver
    d::Int                      # Dimension
    q::Int                      # Order
    dm                          # Dynamics Model
    mm                          # Measurement Model
    sigma_estimator
end

Base.@kwdef mutable struct SolverCache
    t::Real                     # Current time
    dt::Real                    # Current stepsize
    x::Gaussian                 # Current state
end


"""Compute the derivative df/dt(y,t), making use of dy/dt=f(y,t)"""
function get_derivative(f, d)
    dfdy(y, t) = d == 1 ?
        ForwardDiff.derivative((y) -> f(y, t), y) :
        ForwardDiff.jacobian((y) -> f(y, t), y)
    dfdt(y, t) = ForwardDiff.derivative((t) -> f(y, t), t)
    df(y, t) = dfdy(y, t) * f(y, t) + dfdt(y, t)
    return df
end

"""Compute q derivatives of f; Output includes f itself"""
function get_derivatives(f, d, q)
    out = Any[f]
    if q > 1
        for order in 2:q
            push!(out, get_derivative(out[end], d))
        end
    end
    return out
end

function initialize(;ivp, q, dt, σ, method, sigmarule, initialize_derivatives)
    h = dt
    d = length(ivp.u0)
    f(x, t) = ivp.f(x, ivp.p, t)


    # Initialize SSM
    dm = ibm(q, d; σ=σ)
    if method == :ekf0
        mm = ekf0_measurement_model(d, q, ivp)
    elseif method == :ekf1
        mm = ekf1_measurement_model(d, q, ivp)
    else
        throw(Error("method argument not in [:ekf0, :ekf1]"))
    end


    # Initialize problem
    t_0, T = ivp.tspan
    x_0 = ivp.u0

    initialize_derivatives = isnothing(initialize_derivatives) ?
        q <= 3 : initialize_derivatives
    if initialize_derivatives
        derivatives = get_derivatives(f, d, q)
        m_0 = vcat(x_0, [_f(x_0, t_0) for _f in derivatives]...)
    else
        m_0 = [x_0; f(x_0, t_0); zeros(d*(q-1))]
    end
    P_0 = diagm(0 => [zeros(d); ones(d*q)] .+ 1e-16)
    initial_state = Gaussian(m_0, P_0)

    # sigmarules = Dict(
    #     :mle => sigma_mle,
    #     :mle_weighted => sigma_mle_weighted,
    #     :map => sigma_map,
    #     :schober16 => schober16_sigma
    # )
    # sigma_estimator = sigmarules[sigmarule]
    sigma_estimator = sigmarule


    return (Solver(;d=d, q=q, dm=dm, mm=mm,
                   sigma_estimator=sigma_estimator,
            SolverCache(;t=t_0, x=initial_state, dt=h))
end



function prob_solve(ivp, dt;
                    steprule=:constant,
                    sigmarule=MLESigma(),
                    method=:ekf0,
                    q=1, σ=1,
                    progressbar=false,
                    abstol=1e-6,
                    reltol=1e-3,
                    maxiters=1e5,
                    sigma_running=0,
                    smoothed=false,
                    initialize_derivatives=nothing,
                    )
    # Initialize problem
    t_0, T = ivp.tspan
    solver, cache = initialize(ivp=ivp, q=q, dt=dt, σ=σ,
                               method=method,
                               sigmarule=sigmarule,
                               initialize_derivatives=initialize_derivatives)
    sol = StructArray([StateBelief(cache.t, cache.x)])
    proposals = []
    retcode = :Default

    # Filtering
    steprules = Dict(
        :constant => constant_steprule(),
        :pvalue => pvalue_steprule(0.05),
        :baseline => classic_steprule(abstol, reltol),
        :measurement_error => measurement_error_steprule(),
        :measurement_scaling => measurement_scaling_steprule(),
        :schober16 => schober16_steprule(),
    )
    steprule = steprules[steprule]


    if progressbar pbar_update, pbar_close = make_progressbar(0.1) end
    iter = 0
    while cache.t < T
        if progressbar pbar_update(fraction=(cache.t-t_0)/(T-t_0)) end

        # Here happens the main "work"
        proposal = predict_update(solver, cache)

        accept, dt_proposal = steprule(solver, cache, proposal, proposals)
        push!(proposals, (proposal..., accept=accept, dt=cache.dt))
        cache.dt = min(dt_proposal, T-cache.t)

        if accept
            push!(sol, StateBelief(proposal.t, proposal.filter_estimate))
            cache.x = proposal.filter_estimate
            cache.t = proposal.t
        end

        iter += 1
        if iter >= maxiters
            break
            retcode = :MaxIters
        end
    end
    if progressbar pbar_close() end

    # Smoothing
    if smoothed
        sol = smooth(sol, solver, proposals)
    end

    # Calibration
    σ² = static_sigma_estimation(solver.sigma_estimator, solver, proposals)
    if σ² != 1
        for s in sol
            s.x.Σ *= σ²
        end
        for p in proposals
            p.measurement.Σ *= σ²
            p.prediction.Σ *= σ²
            p.filter_estimate.Σ *= σ²
        end
    end

    return (
        prob=ivp,
        solver=solver,
        t=sol.t,
        u=StructArray(sol.x),
        sol=sol,
        proposals=proposals,
        retcode=retcode,
    )
end
