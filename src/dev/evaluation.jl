function add_chi2_statistic!(sol::ProbNumODE.ProbODESolution, sol2::DiffEqBase.AbstractODESolution)
    @assert sol2.dense
    densetimes = collect(range(sol.t[1],stop=sol.t[end],length=100))

    interp_pu = sol.p(densetimes)[2:end]
    interp_analytic = sol2(densetimes)[2:end]

    diffs = interp_pu.μ - interp_analytic.u
    xi2_stats = [r' * inv(cov) * r for (r, cov) in zip(diffs, interp_pu.Σ)]
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

function MyWorkPrecision(prob, alg::AbstractODEFilter, abstols, reltols, name, appxsol, dts=nothing;
                         print_tols=true, numruns=20, seconds=2, kwargs...)
    N = length(abstols)
    errors = []
    costs = []

    for i in 1:N
        if dts === nothing
            print_tols && println("$i: alg=$alg; abstol=$(abstols[i]); reltol=$(reltols[i])")
            sol_fun = () -> solve(prob, alg, abstol=abstols[i], reltol=reltols[i],
                                  kwargs...)
        else
            @assert length(dts) == length(abstols)
            print_tols && println("$i: alg=$alg; abstol=$(abstols[i]); reltol=$(reltols[i]); dt=$(dts[i])")
            sol_fun = () -> solve(prob, alg, abstol=abstols[i], reltol=reltols[i], dt=dts[i],
                                  kwargs...)
        end

        sol = sol_fun()
        sol = appxtrue(sol, appxsol)
        add_chi2_statistic!(sol, appxsol)
        push!(errors, sol.errors)

        benchmark_f = () -> @elapsed sol_fun()
        time = benchmark_f()
        if time < seconds
            time = mapreduce(j -> benchmark_f(), min, 2:numruns; init = time)
        end
        nf = sol.destats.nf
        nf_and_jac = sol.destats.nf + sol.destats.njacs
        push!(costs, Dict(
            :nf => nf,
            :num_evals => nf_and_jac,
            :time => time,
        ))
    end
    # dict_to_namedtuple(d::Dict) = NamedTuple{Tuple(keys(d))}(values(d))
    # costs = StructArray(map(dict_to_namedtuple, costs))
    # errors = StructArray(map(dict_to_namedtuple, errors))

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
