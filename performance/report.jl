# Builds performance/results/report.html from the results in performance/results/julia-*/.
#
#     julia performance/report.jl             # everything, for local use
#     julia performance/report.jl --publish   # docs/src/performance_report.html, without the working tree
#
# The page embeds the raw AirspeedVelocity results and the time-to-first-solve results, and
# renders them with the script in performance/report_template.html.

using Dates: Dates
using TOML: TOML

const RESULTS = joinpath(@__DIR__, "results")
const TEMPLATE = joinpath(@__DIR__, "report_template.html")

# What each benchmark of performance/benchmarks.jl runs, for the labels of the report and for its
# summary by factor. Keep in sync with benchmarks.jl.
#   size: problems are "small" (d ≤ 3) or "large" (d ≥ 14)
#   structure: the covariance structure the solver uses, see `choose_covariance_structure`
problem(key, d, dt, steps, note; extra="") = Dict(
    "key" => key, "d" => d, "dt" => dt, "steps" => steps, "note" => note,
    "extra" => extra,
    "size" => d <= 3 ? "small" : "large")
const PROBLEMS = [
    problem(
        "lotkavolterra",
        2,
        "0.01",
        1000,
        "small non-stiff problem, all solver variants",
    ),
    problem(
        "vanderpol",
        2,
        "1e-3",
        6300,
        "μ = 10, mildly stiff; EK1 against the exponential integrators",
    ),
    problem("pleiades", 28, "0.01", 300, "medium size, non-stiff"),
    problem(
        "pleiades_secondorder",
        14,
        "0.01",
        300,
        "the same problem as a SecondOrderODEProblem",
    ),
    problem(
        "lorenz96",
        50,
        "0.01",
        100,
        "largest problem, where the covariance structure matters most",
    ),
    problem("rober_massmatrix", 3, "1e-5", 1000, "DAE with a singular mass matrix"),
    problem("lotkavolterra_data", 2, "0.01", 1000, "conditioning on 100 data points";
        extra="callback=DataUpdateCallback(data; observation_noise_cov=1e-2)"),
    problem("oscillator_manifold", 2, "0.01", 1000, "energy conservation in every step";
        extra="callback=ManifoldUpdate(E)"),
]
solver(base, variant, structure, call) =
    Dict("base" => base, "variant" => variant, "structure" => structure, "call" => call)
const SOLVERS = Dict(
    "EK0" => solver("EK0", "", "Kronecker", "EK0(smooth=S)"),
    "EK0_DynamicMVDiffusion" => solver("EK0", "DynamicMVDiffusion", "block-diagonal",
        "EK0(smooth=S, diffusionmodel=DynamicMVDiffusion())"),
    "EK0_FixedMVDiffusion" => solver("EK0", "FixedMVDiffusion", "block-diagonal",
        "EK0(smooth=S, diffusionmodel=FixedMVDiffusion())"),
    "DiagonalEK1" =>
        solver("DiagonalEK1", "", "block-diagonal", "DiagonalEK1(smooth=S)"),
    "MatfreeEK1" => solver("MatfreeEK1", "", "block-diagonal", "MatfreeEK1(smooth=S)"),
    "EK1" => solver("EK1", "", "dense", "EK1(smooth=S)"),
    "EK1_IOUP" =>
        solver("EK1", "IOUP prior", "dense", "EK1(smooth=S, prior=IOUP(3, -1.0))"),
    "EK1_Matern" =>
        solver("EK1", "Matern prior", "dense", "EK1(smooth=S, prior=Matern(3, 1.0))"),
    "ExpEK" => solver("ExpEK", "", "dense", "ExpEK(L=-1.0, smooth=S)"),
    "RosenbrockExpEK" =>
        solver("RosenbrockExpEK", "", "dense", "RosenbrockExpEK(smooth=S)"),
)

json(x::AbstractString) = "\"" * escape_string(x) * "\""
json(x::Real) = isfinite(x) ? string(x) : "null"
json(x::AbstractVector) = "[" * join(json.(x), ",") * "]"
json(x::AbstractDict) =
    "{" * join((json(string(k)) * ":" * json(v) for (k, v) in x), ",") * "}"
json(::Nothing) = "null"

function revision_order(rev)
    v = tryparse(VersionNumber, rev)
    v !== nothing && return (0, v, rev)
    rev == "main" && return (1, v"0", rev)
    rev == "dirty" && return (3, v"0", rev)
    return (2, v"0", rev)
end

# The date of the revision itself, with the commit hash for branches.
function revision_date(rev)
    rev == "dirty" && return ""
    out = try
        readchomp(`git -C $(dirname(@__DIR__)) log -1 --format=%cs\ %h $rev --`)
    catch
        return ""
    end
    date, hash = split(out)
    return tryparse(VersionNumber, rev) === nothing ? "$date $hash" : date
end

# The CPU and the thread count of the last run logged in `dir`. The host name is not published.
function last_machine(dir)
    machine = Dict{String,String}()
    file = joinpath(dir, "runs.log")
    isfile(file) || return machine
    for line in eachline(file)
        m = match(r" cpu=(.*) julia version \S+ threads=(\S+)$", line)
        m === nothing && continue
        machine["cpu"], machine["threads"] = m[1], m[2]
    end
    return machine
end

# The files `<prefix>_ProbNumDiffEq@<rev>.<ext>` in `dir`, as pairs `rev => path`.
# AirspeedVelocity and ttfs.jl both write a "/" in a revision as "_", so a revision has
# the same name in both.
function revision_files(dir, prefix, ext)
    matches = match.(Regex("^$(prefix)_ProbNumDiffEq@(.*)\\.$ext\$"), readdir(dir))
    return [m[1] => joinpath(dir, m.match) for m in matches if m !== nothing]
end

# The report shows at most MAX_SAMPLES solves per benchmark and revision, so only those are embedded,
# together with the statistics of all of them. This shrinks the page by a factor of ten.
# `time_to_load` is left as it is: its first sample is not a fresh process, and the report drops it.
const MAX_SAMPLES = 100
function quantile_sorted(v, q)
    pos = (length(v) - 1) * q
    lo, hi = floor(Int, pos), ceil(Int, pos)
    return v[lo+1] + (v[hi+1] - v[lo+1]) * (pos - lo)
end
function shrink(results::AbstractString)
    results = replace(results, r"\"gctimes\":\[[^\]]*\]" => "\"gctimes\":[]")
    return replace(
        results,
        r"(?<!\"time_to_load\":\{)\"times\":\[[^\]]*\]" => function (m)
            times = parse.(Float64, split(m[10:(end-1)], ','))
            n = length(times)
            n <= MAX_SAMPLES && return m
            v = sort(times)
            stats = Dict("n" => n, "min" => v[1], "max" => v[end],
                "q1" => quantile_sorted(v, 0.25),
                "median" => quantile_sorted(v, 0.5),
                "q3" => quantile_sorted(v, 0.75), "p98" => quantile_sorted(v, 0.98),
            )
            sample = round.(Int, times[1:cld(n, MAX_SAMPLES):end])
            return "\"times\":" * json(sample) * ",\"stats\":" * json(stats)
        end,
    )
end

function main(; publish=false)
    dirs = filter(
        d -> startswith(d, "julia-") && isdir(joinpath(RESULTS, d)),
        readdir(RESULTS),
    )
    julia_version(d) = chop(d; head=length("julia-"), tail=0)
    sort!(dirs; by=VersionNumber ∘ julia_version)
    revisions, results, ttfs, machine =
        Dict{String,Any}[], String[], Dict{String,Any}(), Dict{String,String}()
    for d in dirs
        julia, dir = julia_version(d), joinpath(RESULTS, d)
        merge!(machine, last_machine(dir))
        results_files = revision_files(dir, "results", "json")
        for (r, file) in sort(results_files; by=revision_order ∘ first)
            publish && r == "dirty" && continue
            id = "$r@$julia"
            push!(
                revisions,
                Dict("id" => id, "name" => r, "julia" => julia, "date" => revision_date(r)),
            )
            push!(results, json(id) * ":" * shrink(read(file, String)))
        end
        for (r, file) in revision_files(dir, "ttfs", "toml")
            publish && r == "dirty" && continue
            ttfs["$r@$julia"] = TOML.parsefile(file)
        end
    end
    isempty(revisions) &&
        error("No results in $RESULTS; run `just perf` or `just perf-history` first.")
    releases = filter(
        r ->
            tryparse(VersionNumber, r["name"]) !== nothing &&
            r["julia"] == last(revisions)["julia"],
        revisions)
    data =
        "{" *
        join(
            [
                "\"generated\":" * json(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM")),
                "\"machine\":" * json(machine),
                "\"revisions\":" * json(revisions),
                "\"default_baseline\":" *
                json(isempty(releases) ? last(revisions)["id"] : last(releases)["id"]),
                "\"problems\":" * json(PROBLEMS),
                "\"solvers\":" * json(SOLVERS),
                "\"ttfs\":" * json(ttfs),
                "\"results\":{" * join(results, ",") * "}",
            ], ",") * "}"

    # The copy in the docs links back to its docs page, docs/src/performance.md
    back = publish ? "<a class=\"back\" href=\"performance/\">← Documentation</a>" : ""
    page =
        replace(read(TEMPLATE, String), "__BENCH_DATA__" => data, "__BACK_LINK__" => back)
    # The template starts with what belongs into the head: title, meta, link and style
    head_end = last(findfirst("</style>", page))
    head, body = page[1:head_end], lstrip(page[(head_end+1):end])
    out =
        publish ? joinpath(dirname(@__DIR__), "docs", "src", "performance_report.html") :
        joinpath(RESULTS, "report.html")
    write(
        out,
        """
 <!doctype html>
 <html lang="en">
 <head>
 <meta charset="utf-8">
 <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
 $head
 </head>
 <body>
 $body
 </body>
 </html>
 """,
    )
    println("Wrote $out")
    return out
end

main(; publish="--publish" in ARGS)
