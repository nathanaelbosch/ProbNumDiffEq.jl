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


function plot_stepsizes!(p, sol; rejections=true, argv...)
    stepsizes = sol.t[2:end-1] - sol.t[1:end-2]
    if all([h ≈ stepsizes[1] for h in stepsizes])
        stepsizes .= stepsizes[1]
    end
    # @show argv
    pargv = (yscale=:log10,
             # marker=:x,
             label="h (left)",
             ylabel="h",
             legend=:bottomleft,
             argv...)
    plot!(p, sol.t[2:end-1], stepsizes; pargv...)
    rejected = filter(p->!p.accept, sol.proposals)
    if rejections && length(rejected) > 0
        t = [p.t for p in rejected]
        dt = [p.dt for p in rejected]
        scatter!(p, t, dt,
                 marker=:x, color=:black, label="rejected")
    end
    return p
end
plot_stepsizes(sol; argv...) = plot_stepsizes!(plot(), sol; argv...)


function plot_analytic_solution!(p, sol, analytic)
    ts = sol.t
    highres_ts = range(ts[1], ts[end], length=1000)
    plot!(p, highres_ts, hcat(analytic.(highres_ts)...)', label="True solution")
end


function plot_errors!(p, sol, analytic; c=1.96, argv...)
    ts = sol.t

    f_est = [m[1:sol.solver.d] for m in sol.u.μ]
    if sol.solver.d == 1
        f_est = hcat(f_est...)'
    end
    true_sol = analytic.(ts)
    diffs = true_sol .- f_est

    pargv = (yscale=:log10,
             # marker=:x,
             label="Global Error",
             ylabel="function error",
             argv...)
    plot!(p, ts[2:end], norm.(diffs, 2)[2:end];
          pargv...)

    # local_errors = norm.([(diffs[i] - diffs[i-1]) for i in 2:length(diffs)], 2)
    # plot!(p_err, ts[2:end], local_errors, marker=:x, label="Local Error")

    # stds = [sqrt.(diag(P)[1:sol.solver.d]) for P in sol.u.Σ]
    # plot!(p, ts[2:end], norm.(c.*stds[2:end], 2);
    #       linestyle=:dash,
    #       (argv..., labels="$c*std")...)

    return p
end
plot_errors(sol, analytic; argv...) = plot_errors!(plot(), sol, analytic; argv...)


stack(x) = copy(reduce(hcat, x)')
function plot_calibration!(p, sol, analytic; label="")
    ts = sol.t
    d = sol.solver.d

    f_est = [m[1:d] for m in sol.u.μ]
    f_covs = [u.Σ[1:d, 1:d] for u in sol.u]
    if sol.solver.d == 1
        f_est = stack(f_est)
    end
    true_sol = analytic.(ts)
    diffs = true_sol .- f_est

    sugg = [d' * inv(C) * d for (d, C) in zip(diffs, f_covs)]
    plot!(p, ts[2:end], sugg[2:end],
          label=label,
          yscale=:log10,
          )
    return p
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



function plot_sigmas!(p, sol)
    # Plot Sigmas
    accepted_proposals = [p for p in sol.proposals if p.accept]
    ts = [p.t for p in accepted_proposals]
    σ²_dynamic = [p.σ²
                  # dynamic_sigma_estimation(sol.solver.sigma_estimator; p...)
                  for p in accepted_proposals]
    if !all(σ²_dynamic .== 1)
        σ² = σ²_dynamic
    else
        σ²_static = [static_sigma_estimation(sol.solver.sigma_estimator, sol.solver, accepted_proposals[1:i])
                     for i in 1:length(accepted_proposals)]
        σ² = σ²_static
    end
    plot!(p, ts, σ²,
          label="σ² (right)", ylabel="σ²", color=2,
          marker=:x, yscale=:log10, legend=:bottomright)
end


function hairer_plot(sol; analytic=nothing, tol=nothing, f_ylims=nothing, title=nothing)
    analytic = isnothing(analytic) ? solve(sol.prob) : analytic

    p_f = plot_solution(sol)
    plot!(p_f, title=title, ylabel="f(t)")
    plot_analytic_solution!(p_f, sol, analytic)

    p_h = plot_stepsizes(sol)
    plot_sigmas!(twinx(p_h), sol)

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
