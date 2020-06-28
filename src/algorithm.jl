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

    A(h) = UpperTriangular(exp(F*(h)))

    function Q(h)
        function _transdiff_ibm_element(row, col)
            idx = 2 * q + 1 - row - col
            fact_rw = factorial(q - row)
            fact_cl = factorial(q - col)

            return σ ^ 2 * (h ^ idx) / (idx * fact_rw * fact_cl)
        end

        qh_1d = Symmetric([_transdiff_ibm_element(row, col) for col in 0:q, row in 0:q])
        I_d = diagm(0 => ones(d))
        return kron(qh_1d, I_d)
    end

    return (A=A, Q=Q, q=q, d=d)
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
function ekf0_measurement_model(d, q, f, p)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    h(m, t) = H_1*m - f(H_0*m, p, t)
    H(m, t) = H_1
    return (h=h, H=H, R=R)
end
function ekf0_measurement_model(d, q, ivp)
    return ekf0_measurement_model(d, q, ivp.f, ivp.p)
end


# Measurement Model
function ekf1_measurement_model(d, q, f, p, kwargs)
    H_0 = kron([i==1 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    H_1 = kron([i==2 ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    R = zeros(d, d)

    Jf(u, p, t) = (:jac in keys(kwargs)) ?
        kwargs[:jac](u, p, t) :
        ForwardDiff.jacobian(_u -> f(_u, p, t), u)
    h(m, t) = H_1*m - f(H_0*m, p, t)
    H(m, t) = H_1 - Jf(H_0*m, p, t) * H_0
    return (h=h, H=H, R=R)
end
ekf1_measurement_model(d, q, ivp) =
    ekf1_measurement_model(d, q, ivp.f, ivp.p, ivp.kwargs)


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
