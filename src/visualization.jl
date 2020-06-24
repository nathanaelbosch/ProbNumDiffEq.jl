using LinearAlgebra
using Plots
using DifferentialEquations  # Used to estimate the "correct" ODE solution
pyplot()

function plot_solution(sol, derivative=0; true_solution=nothing, stdfac=1.96)
    stds = hcat([sqrt.(diag(M)) for M in sol.u.Σ]...)'
    d = sol.solver.d

    # Plot the results
    means = hcat(sol.u.μ...)'
    p1 = plot(sol.t, means[:, 1:d],
              ribbon=stdfac*stds[:, 1:d],
              marker=:x,
              title="\$f(t)\$", label="Filter Estimate", xlabel="t")
    if !(isnothing(true_solution))
        plot!(p1, range(sol.t[1], sol.t[end], length=1000), true_solution, label="True solution")
    end
    return p1
end


function plot_stepsizes(sol)
    stepsizes = sol.t[2:end-1] - sol.t[1:end-2]
    if all([h ≈ stepsizes[1] for h in stepsizes])
        stepsizes .= stepsizes[1]
    end
    p_h = plot(sol.t[2:end-1],
               stepsizes,
               yscale=:log10,
               marker=:x, label="h (left)", ylabel="h",
               legend=:bottomleft,
               )
    rejected = filter(p->!p.accept, sol.proposals)
    if length(rejected) > 0
        t = [p.t for p in rejected]
        dt = [p.dt for p in rejected]
        scatter!(p_h, t, dt,
                 marker=:x, color=:black, label="rejected")
    end
    return p_h
end


function plot_analytic_solution!(p, sol, analytic)
    ts = sol.t
    highres_ts = range(ts[1], ts[end], length=1000)
    plot!(p, highres_ts, hcat(analytic.(highres_ts)...)', label="True solution")
end


function plot_errors(sol, analytic=nothing)
    ts = sol.t

    f_est = [m[1:sol.solver.d] for m in sol.u.μ]
    if sol.solver.d == 1
        f_est = hcat(f_est...)'
    end
    true_sol = analytic.(ts)
    diffs = true_sol .- f_est
    p_err = plot(ts[2:end], norm.(diffs, 2)[2:end],
                 yscale=:log10, marker=:x,
                 xlabel="t", label="Global Error", ylabel="function error")

    local_errors = norm.([(diffs[i] - diffs[i-1]) for i in 2:length(diffs)], 2)
    plot!(p_err, ts[2:end], local_errors, marker=:x, label="Local Error")

    stds = [sqrt.(diag(P)[1:sol.solver.d]) for P in sol.u.Σ]
    plot!(p_err, ts[2:end], norm.(stds[2:end], 2), marker=:x, label="Output std")

    return p_err
end


function plot_residuals(sol)
    accepted_proposals = [p for p in sol.proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    times = [p.t for p in accepted_proposals]
    res = [z.μ' * z.Σ^(-1) * z.μ for z in measurements] ./ sol.solver.d
    yscale = all(res .> 0) ? :log10 : :identity

    p_res = plot(times, res,
                 marker=:x,
                 ylabel="residual",
                 label="residual",
                 yscale=yscale,
                 )
    plot!(p_res, [1], seriestype=:hline, color=3, label=nothing)

    # Plot std of measurements
    msmnt_vars = [diag(m.Σ) for m in measurements]
    msmnt_stds = [norm(sqrt.(S)) for S in msmnt_vars]
    plot!(p_res, times, 3msmnt_stds,
          label="Measurement std",
          marker=:x, color=2)


    # Plot rejected proposals
    rejected_proposals = [p for p in sol.proposals if !p.accept]
    if length(rejected_proposals) > 0
        times = [p.t for p in rejected_proposals]
        measurements = [p.measurement for p in rejected_proposals]
        res = [z.μ' * z.Σ^(-1) * z.μ for z in measurements] ./ sol.solver.d
        steps = [p.dt for p in rejected_proposals]
        # rejected_t, rejected_h, rejected_errors = collect(zip(sol.rejected...))
        scatter!(p_res, times .+ steps, res,
                 marker=:x, color=:black, label="rejected")
    end

    return p_res
end


function hairer_plot(sol; analytic=nothing, tol=nothing, f_ylims=nothing)
    analytic = isnothing(analytic) ? solve(sol.prob) : analytic

    p_f = plot_solution(sol)
    plot_analytic_solution!(p_f, sol, analytic)

    p_h = plot_stepsizes(sol)
    # Plot Sigmas
    # accepted_proposals = [p for p in sol.proposals if p.accept]
    # ts, sigmas = zip([(accepted_proposals[i].t,
    #                    sol.solver.sigma_estimator(sol.solver, accepted_proposals[1:i]))
    #                   for i in 1:length(accepted_proposals)]...)
    # ts, sigmas = collect(ts), collect(sigmas)
    # # σ² = [(sol.solver.sigma_estimator(accepted_proposals[1:i]) for i in 1:length(accepted_proposals)]
    # plot!(twinx(p_h), ts, sigmas,
    #       label="σ² (right)", ylabel="σ²", color=2,
    #       marker=marker, yscale=:log10, legend=:bottomright)

    # 3: Errors
    p_err = plot_errors(sol, analytic)

    p_res = plot_residuals(sol)

    return plot(
        p_f,
        p_h,
        p_err,
        p_res,
        xlims=(sol.t[1], sol.t[end]),
        layout=(4,1),
        size=(600,800))
end
