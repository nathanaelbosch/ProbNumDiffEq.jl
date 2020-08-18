@userplot Plot_Solution
@recipe function f(ps::Plot_Solution; derivative=0, true_solution=nothing, stdfac=1.96)
    sol = ps.args[1]

    stds = hcat([sqrt.(diag(M)) for M in sol.x.Σ]...)'
    d = sol.solver.constants.d

    # Plot the results
    means = hcat(sol.x.μ...)'

    # return p1
    ribbon --> stdfac*stds[:, 1:d]
    # marker --> :x
    title --> "\$f(t)\$"
    label --> "Filter Estimate"
    xguide --> "t"

    # TODO: Add true solution back in
    # if !(isnothing(true_solution))
    #     plot!(p1, range(sol.t[1], sol.t[end], length=1000), true_solution, label="True solution")
    # end

    return sol.t, means[:, 1:d]
end


@userplot Plot_Stepsizes
@recipe function f(ps::Plot_Stepsizes; rejections=true)
    sol = ps.args[1]

    stepsizes = sol.t[2:end-1] - sol.t[1:end-2]
    if all([h ≈ stepsizes[1] for h in stepsizes])
        stepsizes .= stepsizes[1]
    end

    yscale --> :log10,
    # marker --> :x
    legend --> :bottomleft
    yguide --> "h"

    @series begin
        label --> "h"
        return sol.t[2:end-1], stepsizes
    end

    # TODO Add rejections back in
    rejected = filter(p->!p.accept, sol.proposals)
    if rejections && length(rejected) > 0
        @series begin
            t = [p.t for p in rejected]
            dt = [p.dt for p in rejected]
            marker := :x
            seriescolor := :black
            label := "rejected"
            seriestype := :scatter
            return t, dt
        end
    end
end


@userplot Plot_Analytic_Solution
@recipe function f(pas::Plot_Analytic_Solution)
    sol, analytic = pas.args[1], pas.args[2]
    ts = sol.t
    highres_ts = range(ts[1], ts[end], length=1000)
    label --> "True solution"
    return highres_ts, hcat(analytic.(highres_ts)...)'
end


@userplot Plot_Errors
@recipe function f(pe::Plot_Errors; c=1.96)
    sol, analytic = pe.args[1], pe.args[2]
    ts = sol.t
    d = sol.solver.constants.d

    f_est = map(u -> Measurements.value.(u), sol.u)
    if d == 1
        f_est = hcat(f_est...)'
    end
    true_sol = analytic.(ts)
    diffs = true_sol .- f_est

    @series begin
        yscale --> :log10
        # marker --> :x
        label --> "Global Error"
        yguide --> "function error"
        return ts[2:end], norm.(diffs, 2)[2:end]
    end

    # local_errors = norm.([(diffs[i] - diffs[i-1]) for i in 2:length(diffs)], 2)
    # plot!(p_err, ts[2:end], local_errors, marker=:x, label="Local Error")

    stds = map(u -> Measurements.value.(u), sol.u)
    @series begin
        label --> "$c * std"
        linestyle := :dash
        return ts[2:end], norm.(c.*stds[2:end], 2)
    end
end


@userplot Plot_Calibration
@recipe function f(pc::Plot_Calibration; interval=0.95)
    sol, analytic = pc.args[1], pc.args[2]
    ts = sol.t
    d = sol.solver.constants.d

    accepted_proposals = [p for p in sol.proposals if p.accept]
    filter_estimates = [p.filter_estimate for p in accepted_proposals]
    E0 = sol.solver.constants.E0
    f_est = [E0 * f.μ for f in filter_estimates]
    f_covs = [E0 * f.Σ * E0' for f in filter_estimates]
    if d == 1
        f_est = stack(f_est)
    end
    true_sol = analytic.(ts[2:end])
    diffs = true_sol .- f_est

    sugg = [d' * inv(C) * d for (d, C) in zip(diffs, f_covs)]

    @series begin
        label --> ""
        yscale --> :log10
        return ts[2:end], sugg
    end
    if !isnothing(interval)
        @series begin
            seriestype := :hline
            seriescolor := :black
            linestyle := :dash
            label := "$(interval*100)% Interval"
            p = 1-interval
            return [quantile(Chisq(d), 1-(p/2)), quantile(Chisq(d), p/2)]
        end
    end
end


@userplot Plot_Residuals
@recipe function f(pr::Plot_Residuals)
    sol = pr.args[1]
    d = sol.solver.constants.d

    accepted_proposals = [p for p in sol.proposals if p.accept]
    measurements = [p.measurement for p in accepted_proposals]
    times = [p.t for p in accepted_proposals]
    res = [z.μ' * z.Σ^(-1) * z.μ for z in measurements] ./ d

    yscale = all(res .> 0) ? :log10 : :identity
    yscale --> yscale
    yguide --> "residual"

    @series begin
        # marker --> :x
        label --> "residual"
        return times, res
    end
    @series begin
        seriestype := :hline
        linestyle := :dash
        seriescolor := 3
        label := nothing
        return [1]
    end

    # Plot std of measurements
    msmnt_vars = [diag(m.Σ) for m in measurements]
    msmnt_stds = [norm(sqrt.(S)) for S in msmnt_vars]
    @series begin
        label := "Measurement std"
        seriescolor := 2
        times, 3msmnt_stds
    end

    # Plot rejected proposals
    rejected_proposals = [p for p in sol.proposals if !p.accept]
    if length(rejected_proposals) > 0
        times = [p.t for p in rejected_proposals]
        measurements = [p.measurement for p in rejected_proposals]
        res = [z.μ' * z.Σ^(-1) * z.μ for z in measurements] ./ d
        steps = [p.dt for p in rejected_proposals]
        # rejected_t, rejected_h, rejected_errors = collect(zip(sol.rejected...))
        @series begin
            seriestype := :scatter
            seriescolor := :black
            marker := :x
            label := "rejected"
            return times .+ steps, res
        end
    end
end



@userplot Plot_Sigmas
@recipe function f(ps::Plot_Sigmas)
    sol = ps.args[1]

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

    label --> "σ²"
    yguide --> "σ²"
    # marker --> :x
    yscale --> :log10
    legend --> :bottomright

    return ts, σ²
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
