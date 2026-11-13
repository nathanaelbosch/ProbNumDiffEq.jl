# Runs the performance benchmarks, see performance/README.md.
#
#     julia performance/run.jl                    # the working tree against `main`
#     julia performance/run.jl run REV...         # these revisions, in one session
#     julia performance/run.jl history            # the releases without current results, and `main`
#     julia performance/run.jl report [--publish] # rebuild the report only
#     julia performance/run.jl ttfs REV...        # time to first solve
#     julia performance/run.jl install            # AirspeedVelocity for both Julia versions
#
# Options: `--julia CHANNEL` (a juliaup channel, default `release`) for `run`, and
# `--threads N` (default 1) for `run`, `history` and `ttfs`.

using Dates: Dates
using TOML: TOML

const REPO = dirname(@__DIR__)
const SUITE = joinpath(@__DIR__, "benchmarks.jl")

# Julia versions as juliaup channels, and the releases benchmarked on each.
# Old releases only install on an older Julia; v0.16.5 and v0.16.6 run on both and link the two.
const JULIA = "release"
const RELEASES = ["v0.16.5", "v0.16.6", "v0.17.1", "v0.18.0"]
const OLD_JULIA = "1.10"
const OLD_RELEASES =
    ["v0.11.2", "v0.12.1", "v0.13.1", "v0.14.0", "v0.15.0", "v0.16.5", "v0.16.6"]

git(args...) = readchomp(Cmd(`git -C $REPO $args`))
results_dir(channel) = joinpath(
    @__DIR__, "results",
    "julia-" * readchomp(`julia +$channel --startup-file=no -e 'print(VERSION)'`))
# AirspeedVelocity writes a "/" in a revision as "_"
meta_file(dir, rev) = joinpath(dir, "meta_ProbNumDiffEq@$(replace(rev, "/" => "_")).toml")
suite_version() = first(git("hash-object", SUITE), 12)

# Benchmarks one revision (`dirty` is the working tree), and records next to its results what
# was measured: the commit, the version of the suite, and the machine.
function run_revision(rev; channel=JULIA, threads=1)
    dir = mkpath(results_dir(channel))
    commit, date = if rev == "dirty"
        git("rev-parse", "--short", "HEAD") * " + changes", string(Dates.today())
    else
        git("rev-parse", "--short", "$rev^{commit}"), git("log", "-1", "--format=%cs", rev)
    end
    suite = suite_version()
    asv = "using AirspeedVelocity.BenchPkg: BenchPkg; exit(BenchPkg.command_main())"
    cmd = `julia --startup-file=no --project=@airspeedvelocity-$channel -e $asv --
        --script $SUITE --output-dir $dir --rev $rev
        --exeflags=$("-t $threads --heap-size-hint=32G")`
    # The benchmark processes inherit the Julia version through JULIAUP_CHANNEL. The working
    # tree runs strictly, so that a benchmark it breaks fails instead of being skipped.
    env = ["JULIAUP_CHANNEL" => channel, "JULIA_NUM_PRECOMPILE_TASKS" => "16"]
    rev == "dirty" && push!(env, "PNDE_BENCH_STRICT" => "1")
    run(Cmd(addenv(cmd, env...); dir=REPO))
    meta = Dict(
        "commit" => commit, "date" => date, "suite" => suite,
        "threads" => string(threads),
        "cpu" => strip(Sys.cpu_info()[1].model), "measured" => string(Dates.now()))
    open(io -> TOML.print(io, meta), meta_file(dir, rev), "w")
    return nothing
end

# Whether a revision has results from the current version of the suite
function is_current(rev, channel)
    file = meta_file(results_dir(channel), rev)
    return isfile(file) && get(TOML.parsefile(file), "suite", "") == suite_version()
end

report(args...) =
    run(`$(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, "report.jl")) $args`)

function history(; threads)
    failed = String[]
    for (channel, releases) in ((JULIA, RELEASES), (OLD_JULIA, OLD_RELEASES)),
        rev in releases

        is_current(rev, channel) && continue
        try
            run_revision(rev; channel, threads)
        catch e
            e isa ProcessFailedException || rethrow()
            push!(failed, "$rev on Julia $channel")
        end
    end
    run_revision("main"; threads)
    report()
    isempty(failed) || error("Benchmarking failed for: " * join(failed, ", "))
    return nothing
end

function main(args)
    args = copy(args)
    option(name, default) =
        (i = findfirst(==(name), args)) === nothing ? default :
        (value=args[i+1]; deleteat!(args, i:(i+1)); value)
    threads = parse(Int, option("--threads", "1"))
    channel = option("--julia", JULIA)
    command = isempty(args) ? "run" : popfirst!(args)
    if command == "run"
        revs = isempty(args) ? ["main", "dirty"] : args
        foreach(rev -> run_revision(rev; channel, threads), revs)
        report()
    elseif command == "history"
        history(; threads)
    elseif command == "report"
        report(args...)
    elseif command == "ttfs"
        ttfs = joinpath(@__DIR__, "ttfs.jl")
        run(
            `julia +$JULIA --startup-file=no $ttfs --exeflags $("-t $threads --heap-size-hint=32G") $args`,
        )
        report()
    elseif command == "install"
        for c in (JULIA, OLD_JULIA)
            run(
                `julia +$c --project=@airspeedvelocity-$c -e 'using Pkg; Pkg.add("AirspeedVelocity")'`,
            )
        end
    else
        error("Unknown command `$command`; see the top of performance/run.jl.")
    end
    return nothing
end

main(ARGS)
