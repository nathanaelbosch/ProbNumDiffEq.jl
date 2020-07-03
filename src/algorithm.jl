"""
    Gaussian(μ::AbstractVector, Σ::AbstractMatrix)

Multivariate Gaussian distribution ``\\mathcal{N}(\\mu, \\Sigma)``.

**Note:* There is currently no additional functionality implemented.
In the future we might instead use Distributions.jl.
"""
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
copy(g::Gaussian) = Gaussian(g.μ, g.Σ)


mutable struct StateBelief
    t::Real
    x::Gaussian
end


# Kalman Filter:
"""
No strict separation between prediction and update, since estimating σ at each step
requires a different order of comutation of the update and predict steps.
"""
function predict_update(integ)
    @unpack dm, mm, x, t, dt = integ
    precond_P = integ.preconditioner.P
    precond_P_inv = integ.preconditioner.P_inv

    t = t + dt

    m, P = x.μ, x.Σ
    A, Q = dm.A(dt), dm.Q(dt)
    A = precond_P * A * precond_P_inv
    Q = Symmetric(precond_P * Q * precond_P')

    m_p = A * m

    # H, R = mm.H(m_p, t), mm.R
    H, R = mm.H(m_p, t) * precond_P_inv, mm.R
    # v = 0 .- mm.h(m_p, t)
    v = 0 .- mm.h(precond_P_inv * m_p, t)

    # Decide if constant sigma or variable sigma
    # σ² = solver.sigma_type == :fixed ? 1 :
    #     solver.sigma_estimator(;H=H, Q=Q, v=v)
    σ² = dynamic_sigma_estimation(integ.sigma_estimator;
                                  H=H, Q=Q, v=v, P=P, A=A, R=R)

    P_p = Symmetric(A*P*A') + σ²*Q
    S = Symmetric(H * P_p * H' + R)
    K = P_p * H' * inv(S)
    m = m_p + K*v
    P = P_p - Symmetric(K*S*K')

    return (t=t,
            prediction=Gaussian(m_p, P_p),
            filter_estimate=Gaussian(m, P),
            measurement=Gaussian(v, S),
            H=H, Q=Q, v=v,
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

function smooth!(sol, proposals, integ)
    precond_P = integ.preconditioner.P
    precond_P_inv = integ.preconditioner.P_inv


    smoothed_solution = StructArray{StateBelief}(undef, length(sol))
    smoothed_solution[end] = sol[end]
    smoothed_solution[1] = sol[1]
    accepted_proposals = [p for p in proposals if p.accept]
    for i in length(smoothed_solution)-1:-1:2
        h = accepted_proposals[i].dt  # step t -> t+1
        h2 = sol[i+1].t - sol[i].t
        @assert h ≈ h2

        @assert accepted_proposals[i].t == sol[i+1].t

        prediction = accepted_proposals[i].prediction  # t+1
        filter_estimate = sol[i].x  # t
        smoothed_estimate = sol[i+1].x # t+1

        A = integ.dm.A(h)
        Q = integ.dm.Q(h)
        A = precond_P * A * precond_P_inv
        Q = Symmetric(precond_P * Q * precond_P')
        sol[i].x = smooth(filter_estimate,
                          prediction,
                          smoothed_estimate,
                          (A=A, Q=Q))
    end
    # return smoothed_solution
end



function preconditioner(expected_stepsize, d, q)
    h = expected_stepsize
    I_d = diagm(0 => ones(d))
    P = Diagonal(kron(Diagonal(h .^ 0:(q+1)), I_d))
    P_inv = Diagonal(kron(Diagonal(1 ./ (h .^ 0:(q+1))), I_d))
    return (P=P, P_inv=P_inv)
    # return (P=I, P_inv=I)
end
function apply_preconditioner!(p, x::Gaussian)
    x.μ .= p.P * x.μ
    x.Σ .= Symmetric(p.P * x.Σ * p.P')
end
function undo_preconditioner!(p, x::Gaussian)
    x.μ .= p.P_inv * x.μ
    x.Σ .= Symmetric(p.P_inv * x.Σ * p.P_inv')
end


# Measurement Model
function measurement_model(kind, d, q, f, p)
    @assert kind in (:ekf0, :ekf1) ("Type of measurement model not in [:ekf0, :ekf1]")
    if kind == :ekf0
        return ekf0_measurement_model(d, q, f, p)
    elseif kind == :ekf1
        return ekf1_measurement_model(d, q, f, p)
    end
end
measurement_model(kind, d, q, ivp) = measurement_model(kind, d, q, ivp.f, ivp.p)

function ekf0_measurement_model(d, q, f, p)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h(m, t) = H_1*m - f(H_0*m, p, t)
    H(m, t) = H_1
    return (h=h, H=H, R=R)
end

function ekf1_measurement_model(d, q, f, p)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h(m, t) = H_1*m - f(H_0*m, p, t)
    Jf = (hasfield(typeof(f), :jac) && !isnothing(f.jac)) ? f.jac :
        (u, p, t) -> ForwardDiff.jacobian(_u -> f(_u, p, t), u)
    H(m, t) = H_1 - Jf(H_0*m, p, t) * H_0

    return (h=h, H=H, R=R)
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

function calibrate!(sol, proposals, integ)
    σ² = static_sigma_estimation(integ.sigma_estimator, integ, proposals)
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
end

function undo_preconditioner!(sol, proposals, integ)
    for s in sol
        undo_preconditioner!(integ.preconditioner, s.x)
    end
    for p in proposals
        undo_preconditioner!(integ.preconditioner, p.prediction)
        undo_preconditioner!(integ.preconditioner, p.filter_estimate)
    end
end
