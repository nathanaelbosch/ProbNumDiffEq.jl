using LinearAlgebra
using Plots
using DifferentialEquations  # Used to estimate the "correct" ODE solution
pyplot()

function plot_solution(sol; true_solution=nothing, true_derivative=nothing, stdfac=1.96)
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
    p2 = plot(sol.t, means[:, d+1:2d],
              ribbon=stdfac*stds[:, d+1:2d], marker=:x,
              title="\$f'(t)\$", label="Filter Estimate", xlabel="t")
    if !(isnothing(true_derivative))
        plot!(p2, range(sol.t[1], sol.t[end], length=1000), true_derivative, label="True solution")
    end
    return p1, p2
end


function hairer_plot(sol; true_sol=nothing, tol=nothing, marker=:x)
    # 1: Function result
    # sol.u.μ
    p_f, p_df = plot_solution(sol)
    plot!(p_f, ylabel="y(t)", xlabel="", title="")


    # 2: Step sizes
    p_h = plot(sol.t[2:end-1],
               sol.t[2:end-1] - sol.t[1:end-2],
               yscale=:log10,
               marker=marker, label="h (left)", ylabel="h",
               legend=:bottomleft,
               )
    rejected = filter(p->!p.accept, sol.proposals)
    if length(rejected) > 0
        t = [p.t for p in rejected]
        dt = [p.dt for p in rejected]
        scatter!(p_h, t, dt,
                 marker=marker, color=:black, label="rejected")
    end
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
    # Get estimate for the true solution from some classic ODE solver
    reference_solution = isnothing(true_sol) ? solve(sol.prob) : true_sol
    ts = sol.t
    highres_ts = range(ts[1], ts[end], length=1000)
    plot!(p_f, highres_ts, hcat(reference_solution.(highres_ts)...)', label="True solution")

    f_est = [m[1:sol.solver.d] for m in sol.u.μ]
    if sol.solver.d == 1
        f_est = hcat(f_est...)'
    end
    true_sol = reference_solution.(ts)
    diffs = true_sol .- f_est
    p_err = plot(ts[2:end], norm.(diffs, 2)[2:end],
                 yscale=:log10, marker=marker,
                 xlabel="t", label="Global Error", ylabel="function error")

    local_errors = norm.([(diffs[i] - diffs[i-1]) for i in 2:length(diffs)], 2)
    plot!(p_err, ts[2:end], local_errors, marker=marker, label="Local Error")

    stds = [sqrt.(diag(P)[1:sol.solver.d]) for P in sol.u.Σ]
    plot!(p_err, ts[2:end], norm.(3*stds[2:end], 2), marker=marker, label="3 * Solput std")


    # # 4: Residuals
    # quadratic_errors = [m.μ' * m.Σ^(-1) * m.μ for m in sol.result.measurements[2:end]] ./ sol.d
    # quadratic_errors = vcat(quadratic_errors...)
    # calibrated_errors = quadratic_errors ./ sol.σ²[1:end-1]

    # yscale = all(calibrated_errors .> 0) ? :log10 : :identity
    # p_res = plot(sol.t, calibrated_errors, marker=:x,
    #              # title="\$E_t\$ (using \$σ^2_{t-1}\$)",
    #              # title="\$E_t := (z_t)^T (σ^2_{t-1} S_t)^{-1} (z_t)\$",
    #              ylabel="residual",
    #              label="residual",
    #              yscale=yscale,
    #              )
    # plot!(p_res, [1], seriestype=:hline, color=3, label=nothing)
    # if length(sol.rejected) > 0
    #     rejected_t, rejected_h, rejected_errors = collect(zip(sol.rejected...))
    #     scatter!(p_res, vcat(rejected_t...), vcat(rejected_errors...),
    #              marker=marker, color=:black, label="rejected")
    # end

    # msmnt_vars = [diag(m.Σ) for m in sol.result.measurements[2:end]] .* sol.σ²[1:end-1]
    # msmnt_stds = [norm(sqrt.(S)) for S in msmnt_vars]
    # plot!(p_res, sol.t[2:end], 3msmnt_stds,
    #       label="3 * Measurement std", marker=marker, color=2)


    # if !isnothing(tol)
    #     plot!(p_err, [tol], seriestype=:hline, color=:black, label="", linestyle=:dot)
    # end

    return plot(
        p_f,
        p_h,
        p_err,
        # p_res,
        xlims=(sol.t[1], sol.t[end]),
        layout=(3,1),
        size=(700,700))
end
