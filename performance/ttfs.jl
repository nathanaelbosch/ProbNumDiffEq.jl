# Time to first solve: `using ProbNumDiffEq` and the first solves in a fresh process.
#
#     julia performance/ttfs.jl [--samples N] [--exeflags FLAGS] REV...
#
# REV is a git revision of the repository or `dirty` for the working tree. For each one
# this installs the package into a temporary environment, precompiles it, and then times
# N fresh Julia processes. Results go to
# performance/results/julia-<version>/ttfs_ProbNumDiffEq@<rev>.toml.

using Pkg: Pkg
using TOML: TOML

const REPO = dirname(@__DIR__)
const RESULTS = joinpath(@__DIR__, "results", "julia-$VERSION")
const NAMES = ("using", "first_solve_EK0", "first_solve_EK1")
# A "/" in a revision becomes "_", as in the result files of AirspeedVelocity, so that
# performance/report.jl finds both under the same name.
result_file(rev) = joinpath(RESULTS, "ttfs_ProbNumDiffEq@$(replace(rev, "/" => "_")).toml")

const MEASURE = raw"""
t_load = @elapsed using ProbNumDiffEq
f(du, u, p, t) = (du[1] = 1.5u[1] - u[1] * u[2]; du[2] = -3u[2] + u[1] * u[2]; nothing)
prob = ODEProblem(f, [1.0, 1.0], (0.0, 1.0))
t_ek0 = @elapsed solve(prob, EK0())
t_ek1 = @elapsed solve(prob, EK1())
println("TTFS ", t_load, " ", t_ek0, " ", t_ek1)
"""

median(x) = (y=sort(x); n=length(y); (y[(n+1)÷2] + y[n÷2+1]) / 2)

function parse_args(args)
    samples, exeflags, revs = 5, "-t 1 --heap-size-hint=32G", String[]
    i = 1
    while i <= length(args)
        if args[i] == "--samples"
            samples = parse(Int, args[i+1])
            i += 2
        elseif args[i] == "--exeflags"
            exeflags = args[i+1]
            i += 2
        else
            push!(revs, args[i])
            i += 1
        end
    end
    isempty(revs) && error("no revision given")
    return samples, split(exeflags), revs
end

function install(rev)
    env = mktempdir()
    julia = Base.julia_cmd()
    spec =
        rev == "dirty" ? "Pkg.develop(path=\"$REPO\")" :
        "Pkg.add(Pkg.PackageSpec(path=\"$REPO\", rev=\"$rev\"))"
    run(`$julia --startup-file=no --project=$env -e "using Pkg; $spec; Pkg.precompile()"`)
    return env
end

function measure(env, exeflags)
    julia = Base.julia_cmd()
    out = read(`$julia --startup-file=no $exeflags --project=$env -e $MEASURE`, String)
    line = only(filter(startswith("TTFS "), split(out, '\n')))
    return parse.(Float64, split(line)[2:end])
end

function main(args)
    samples, exeflags, revs = parse_args(args)
    mkpath(RESULTS)
    for rev in revs
        @info "Time to first solve for $rev"
        env = install(rev)
        measure(env, exeflags)  # the first process after precompilation can be slower
        times = [measure(env, exeflags) for _ in 1:samples]
        result = Dict{String,Any}(
            name => Dict(
                "median" => median(getindex.(times, k)),
                "samples" => getindex.(times, k),
            )
            for (k, name) in enumerate(NAMES)
        )
        result["julia"] = string(VERSION)
        result["exeflags"] = join(exeflags, " ")
        open(io -> TOML.print(io, result), result_file(rev), "w")
    end
    println()
    println(
        rpad("revision", 16),
        lpad("using", 10),
        lpad("1st EK0", 10),
        lpad("1st EK1", 10),
    )
    for rev in revs
        r = TOML.parsefile(result_file(rev))
        cols = (r[k]["median"] for k in NAMES)
        println(rpad(rev, 16), join(lpad.(string.(round.(cols, digits=2), " s"), 10)))
    end
    return nothing
end

main(ARGS)
