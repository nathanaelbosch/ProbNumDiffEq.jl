
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


    smoothed_solution = _copy(sol)
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
        sol[i] = (t=sol[i].t,
                  x=smooth(filter_estimate,
                           prediction,
                           smoothed_estimate,
                           (A=A, Q=Q)))
    end
    # return smoothed_solution
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
