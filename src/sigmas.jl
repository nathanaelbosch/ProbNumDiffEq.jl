function sigma_mle_weighted(solver, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = length(measurements[1].μ)
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    stepsizes = [p.dt for p in accepted_proposals]
    σ² = mean(residuals .* stepsizes)
    return σ²
end

function sigma_mle(solver, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = length(measurements[1].μ)
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    σ² = mean(residuals)
    return σ²
end

function sigma_map(solver, proposals)
    accepted_proposals = [p for p in proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    d = length(measurements[1].μ)
    residuals = [v.μ' * inv(v.Σ) * v.μ for v in measurements] ./ d
    N = length(residuals)

    α, β = 1/2, 1/2
    # prior = InverseGamma(α, β)
    α2, β2 = α + N*d/2, β + 1/2 * (sum(residuals))
    posterior = InverseGamma(α2, β2)
    sigma = mode(posterior)
    return sigma
end

# function sigma_running_average(msmnt_errors, current_error; running=0, argv...)
#     N = min(length(msmnt_errors), running)
#     if N > 0
#         sigma = (sum(msmnt_errors[(end-(N-1)):end]) + current_error) ./ (N+1)
#     else
#         sigma = current_error
#     end
#     return sigma
# end

function schober16_sigma(solver, proposals)
    proposal = proposals[end]
    v = proposal.measurement.μ
    Q = solver.dm.Q(proposal.dt)
    H = solver.mm.H(proposal.prediction.μ, proposal.t)
    sigma = v' * inv(H*Q*H') * v / length(v)
    return sigma
end
function schober16_sigma(;H, Q, v)
    return v' * inv(H*Q*H') * v / length(v)
end
