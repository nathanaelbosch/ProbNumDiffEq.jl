function add_dense_chi2_statistic!(sol::ProbNumODE.ProbODESolution,
                                   sol2::DiffEqBase.AbstractODESolution)
    :Χ² in keys(sol.errors) && error("χ² error is already computed!")

    densetimes = collect(range(sol.t[1],stop=sol.t[end],length=100))
    interp_pu = sol.p(densetimes)[2:end]

    @assert sol2.dense
    interp_analytic = sol2(densetimes)[2:end]
    diffs = interp_pu.μ - interp_analytic.u
    xi2_stats = [r' * inv(cov) * r for (r, cov) in zip(diffs, interp_pu.Σ)]
    sol.errors[:Χ²] = mean(xi2_stats)

    return nothing
end

function add_dense_chi2_statistic!(sol::ProbNumODE.ProbODESolution)
    :Χ² in keys(sol.errors) && error("χ² error is already computed!")

    densetimes = collect(range(sol.t[1],stop=sol.t[end],length=100))
    interp_pu = sol.p(densetimes)

    # Skip all the zero-cov results because they make computation nasty
    means, covs = interp_pu.μ, interp_pu.Σ
    vars = diag.(covs)
    is_zero = all.(stack(vars) .== 0)
    @assert issorted(is_zero, rev=true)
    skip_idx = sum(is_zero) + 1
    means, covs = means[skip_idx:end], covs[skip_idx:end]

    @assert DiffEqBase.has_analytic(sol.prob.f)
    f_analytic(t) = sol.prob.f.analytic(sol.prob.u0, sol.prob.p, t)
    u_analytic = f_analytic.(densetimes)[skip_idx:end]

    # @show interp_pu
    @assert length(means) > 0 "All values are skipped in the computation of the chi2-statistic!"
    # @show size(means), size(u_analytic)
    diffs = means - u_analytic
    xi2_stats = [r' * inv(cov) * r for (r, cov) in zip(diffs, covs)]
    sol.errors[:Χ²] = mean(xi2_stats)
end

function add_discrete_chi2_statistic!(sol::ProbNumODE.ProbODESolution,
                                      sol2::DiffEqBase.AbstractODESolution)
    :χ² in keys(sol.errors) && error("Χ² error is already computed! $(keys(sol.errors))")

    interp_pu = sol.pu[2:end]

    interp_analytic = sol2(sol.t)[2:end]
    diffs = interp_pu.μ - interp_analytic.u
    xi2_stats = [r' * inv(cov) * r for (r, cov) in zip(diffs, interp_pu.Σ)]
    sol.errors[:χ²] = mean(xi2_stats)

    return nothing
end

function add_discrete_chi2_statistic!(sol::ProbNumODE.ProbODESolution)
    :χ² in keys(sol.errors) && error("Χ² error is already computed! $(keys(sol.errors))")

    interp_pu = sol.pu

    # Skip all the zero-cov results because they make computation nasty
    means, covs = interp_pu.μ, interp_pu.Σ
    vars = diag.(covs)
    is_zero = all.(stack(vars) .== 0)
    @assert issorted(is_zero, rev=true)
    skip_idx = sum(is_zero) + 1
    means, covs = means[skip_idx:end], covs[skip_idx:end]

    @assert DiffEqBase.has_analytic(sol.prob.f)
    f_analytic(t) = sol.prob.f.analytic(sol.prob.u0, sol.prob.p, t)
    u_analytic = f_analytic.(sol.t)[skip_idx:end]

    diffs = means - u_analytic
    xi2_stats = [r' * inv(cov) * r for (r, cov) in zip(diffs, covs)]
    sol.errors[:χ²] = mean(xi2_stats)

    return nothing
end


mutable struct MyWorkPrecision
    prob::DiffEqBase.AbstractODEProblem
    abstols::Vector{Real}
    reltols::Vector{Real}
    errors::Vector{Dict}
    costs::Vector{Dict}
    name::String
end


function compute_costs_errors(prob, alg, abstol, reltol, appxsol, dt=nothing;
                              numruns=20, seconds=2, kwargs...)
    if dt === nothing
        sol_fun = () -> solve(prob, alg; abstol=abstol, reltol=reltol,
                              kwargs...)
    else
        sol_fun = () -> solve(prob, alg; abstol=abstol, reltol=reltol, dt=dt,
                              kwargs...)
    end

    sol = sol_fun()
    if DiffEqBase.has_analytic(sol.prob.f)
        @assert :l2 in keys(sol.errors)
        @assert :χ² in keys(sol.errors)
    else
        sol = appxtrue(sol, appxsol)
        add_dense_chi2_statistic!(sol, appxsol)
        add_discrete_chi2_statistic!(sol, appxsol)
    end

    benchmark_f = () -> @elapsed sol_fun()
    time = benchmark_f()
    if time < seconds
        time = mapreduce(j -> benchmark_f(), min, 2:numruns; init = time)
    end
    nf = sol.destats.nf
    nf_and_jac = sol.destats.nf + sol.destats.njacs
    costs = Dict(
        :nf => nf,
        :num_evals => nf_and_jac,
        :time => time,
    )
    return costs, sol.errors
end


function MyWorkPrecision(prob, alg::AbstractODEFilter, abstols, reltols, name, appxsol, dts=nothing;
                         print_tols=true, numruns=20, seconds=2, kwargs...)
    N = length(abstols)
    errors = []
    costs = []

    for i in 1:N
        if dts === nothing
            print_tols && println("$i: alg=$alg; abstol=$(abstols[i]); reltol=$(reltols[i])")
            c, e = compute_costs_errors(prob, alg, abstols[i], reltols[i], appxsol;
                                        numruns=numruns, seconds=seconds, kwargs...)
        else
            print_tols && println("$i: alg=$alg; abstol=$(abstols[i]); reltol=$(reltols[i]); dt=$(dts[i])")
            c, e = compute_costs_errors(prob, alg, abstols[i], reltols[i], appxsol, dts[i];
                                        numruns=numruns, seconds=seconds, kwargs...)
        end

        push!(errors, e)
        push!(costs, c)
    end

    return MyWorkPrecision(prob, abstols, reltols, errors, costs, name)
end



mutable struct MyWorkPrecisionSet
    wps::Vector{MyWorkPrecision}
end

function MyWorkPrecisionSet(prob, abstols, reltols, setups, names, appxsol, dts=nothing;
                            print_names=true, kwargs...)
    N = length(setups)
    wps = Vector{MyWorkPrecision}(undef,N)

    for i in 1:N
        print_names && println(names[i])
        wps[i] = MyWorkPrecision(prob, setups[i][:alg], abstols, reltols, names[i], appxsol, dts;
                                 kwargs..., setups[i]...)
    end
    return MyWorkPrecisionSet(wps)
end


@recipe function f(wp::MyWorkPrecision; error=:L2, cost=:time)
    x = [c[cost] for c in wp.costs]
    y = [e[error] for e in wp.errors]

    seriestype --> :path
    label -->  wp.name
    linewidth --> 3
    yguide --> "Error: $error"
    xguide --> "Cost: $cost"
    xscale --> :log10
    yscale --> :log10
    marker --> :auto
    @series begin
        return x, y
    end

    if error == :χ²
        @series begin
            seriestype := :hline
            linestyle := :dash
            seriescolor := :black
            return [1]
        end
    end
end


@recipe function f(wps::MyWorkPrecisionSet; error=:L2, cost=:time)
    xs = Vector{Any}(undef,0)
    ys = Vector{Any}(undef,0)
    for wp in wps.wps
        push!(xs, [c[cost] for c in wp.costs])
        push!(ys, [e[error] for e in wp.errors])
    end
    names = [wp.name for wp in wps.wps]
    label --> reshape(names, 1, length(names))

    seriestype --> :path
    linewidth --> 3
    yguide --> "Error: $error"
    xguide --> "Cost: $cost"
    xscale --> :log10
    yscale --> :log10
    marker --> :auto
    @series begin
        return xs, ys
    end

    if error == :χ²
        @series begin
            seriestype := :hline
            linewidth := 1
            linestyle := :dash
            seriescolor := :black
            return [1]
        end
    end
end
