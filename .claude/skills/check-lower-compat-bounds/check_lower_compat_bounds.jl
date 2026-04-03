"""
    check_lower_compat_bounds.jl

Test whether the lower compat bounds in Project.toml are still resolvable.

For each dependency, creates a fresh temporary environment, dev's the current
package, and tries to add the dependency at its lower compat bound. If resolution
fails, binary searches for the true minimum resolvable version.

Usage:
    julia .claude/skills/check-lower-compat-bounds/check_lower_compat_bounds.jl
    julia .claude/skills/check-lower-compat-bounds/check_lower_compat_bounds.jl --apply

The first form runs all checks, prints a report, and saves the suggested updates
to a file. The second form applies the saved suggestions to Project.toml without
re-running the checks.
"""

using Pkg, TOML, Printf

# ============================================================================
# Registry helpers
# ============================================================================

function available_versions(pkg_name::String)
    for reg in Pkg.Registry.reachable_registries()
        for (_, pkg) in reg.pkgs
            if pkg.name == pkg_name
                info = Pkg.Registry.registry_info(pkg)
                return sort(collect(keys(info.version_info)))
            end
        end
    end
    return VersionNumber[]
end

# ============================================================================
# Resolution testing
# ============================================================================

const PROJECT_DIR = dirname(dirname(dirname(dirname(@__FILE__))))

function test_resolve(pkg::String, ver::VersionNumber)::Bool
    dir = mktempdir()
    Pkg.activate(dir; io=devnull)
    try
        Pkg.develop(; path=PROJECT_DIR, io=devnull)
        Pkg.add(; name=pkg, version=string(ver), io=devnull)
        return true
    catch
        return false
    end
end

# ============================================================================
# Compat parsing
# ============================================================================

"""Parse a compat string like "1.2, 3" into a list of lower-bound VersionNumbers."""
function parse_compat_ranges(compat_str::String)
    bounds = VersionNumber[]
    for part in split(compat_str, ",")
        s = strip(part)
        isempty(s) && continue
        # Handle ranges like ">=1.2" or "1.2 - 3" by taking the first version-like token
        m = match(r"(\d+(?:\.\d+)*)", s)
        m === nothing && continue
        push!(bounds, VersionNumber(m.match))
    end
    return bounds
end

"""
    version_range_for_bound(lower::VersionNumber) -> (lo, hi)

Given a compat lower bound like v"1.2.3", return the version range it covers.
Julia compat semantics: "1.2" means [1.2, 2), "0.5" means [0.5, 0.6).
"""
function compat_upper_bound(lower::VersionNumber)
    if lower.major == 0
        return VersionNumber(0, lower.minor + 1, 0)
    else
        return VersionNumber(lower.major + 1, 0, 0)
    end
end

# ============================================================================
# Binary search
# ============================================================================

"""
    find_minimum_resolvable(pkg, versions) -> Union{VersionNumber, Nothing}

Binary search over `versions` (sorted ascending) to find the lowest version that
resolves. Returns nothing if none resolve.
"""
function find_minimum_resolvable(pkg::String, versions::Vector{VersionNumber})
    isempty(versions) && return nothing

    lo, hi = 1, length(versions)

    # Quick check: does the highest version resolve?
    if !test_resolve(pkg, versions[hi])
        return nothing
    end

    # Quick check: does the lowest version resolve?
    if test_resolve(pkg, versions[lo])
        return versions[lo]
    end

    # Binary search: invariant: versions[lo] fails, versions[hi] succeeds
    while hi - lo > 1
        mid = (lo + hi) >> 1
        if test_resolve(pkg, versions[mid])
            hi = mid
        else
            lo = mid
        end
    end

    return versions[hi]
end

# ============================================================================
# Main logic
# ============================================================================

const SKIP_PKGS = Set([
    # Stdlibs
    "LinearAlgebra", "Printf", "Random", "Statistics", "Test",
])

struct CompatResult
    pkg::String
    current_compat::String
    range_lower::VersionNumber
    range_upper::VersionNumber
    installed::Union{VersionNumber,Nothing}
    lower_resolves::Bool
    min_resolvable::Union{VersionNumber,Nothing}
end

function check_all_bounds(project_dir::String)
    project_toml = TOML.parsefile(joinpath(project_dir, "Project.toml"))
    compat_section = get(project_toml, "compat", Dict())

    # Get installed versions from the current environment
    Pkg.activate(project_dir; io=devnull)
    deps_info = Pkg.dependencies()
    project_deps = get(project_toml, "deps", Dict())

    installed_versions = Dict{String,VersionNumber}()
    for (name, uuid_str) in project_deps
        uuid = Base.UUID(uuid_str)
        if haskey(deps_info, uuid)
            installed_versions[name] = deps_info[uuid].version
        end
    end

    results = CompatResult[]

    for (pkg, compat_str) in sort(collect(compat_section))
        pkg in SKIP_PKGS && continue
        pkg == "julia" && continue

        bounds = parse_compat_ranges(compat_str)
        isempty(bounds) && continue
        installed = get(installed_versions, pkg, nothing)

        for lower in bounds
            upper = compat_upper_bound(lower)

            # Get available versions in this compat range
            all_versions = available_versions(pkg)
            range_versions = filter(v -> v >= lower && v < upper, all_versions)

            if isempty(range_versions)
                push!(
                    results,
                    CompatResult(
                        pkg, compat_str, lower, upper, installed,
                        false, nothing,
                    ),
                )
                continue
            end

            print("Testing $pkg $(lower)...");
            flush(stdout)
            if test_resolve(pkg, lower)
                println(" OK");
                flush(stdout)
                push!(
                    results,
                    CompatResult(
                        pkg, compat_str, lower, upper, installed,
                        true, lower,
                    ),
                )
            else
                print(" FAIL, searching...");
                flush(stdout)
                min_ver = find_minimum_resolvable(pkg, range_versions)
                if min_ver !== nothing
                    println(" minimum: $min_ver");
                    flush(stdout)
                else
                    println(" entire range dead");
                    flush(stdout)
                end
                push!(
                    results,
                    CompatResult(
                        pkg, compat_str, lower, upper, installed,
                        false, min_ver,
                    ),
                )
            end
        end
    end

    return results
end

function print_report(results::Vector{CompatResult})
    dead = filter(r -> !r.lower_resolves && r.min_resolvable === nothing, results)
    stale = filter(r -> !r.lower_resolves && r.min_resolvable !== nothing, results)
    ok = filter(r -> r.lower_resolves, results)

    println("\n", "="^70)
    println("COMPAT BOUNDS REPORT")
    println("="^70)

    if !isempty(dead)
        println("\nDEAD RANGES (entire range unresolvable, should be dropped):")
        println("-"^60)
        for r in dead
            println("  $(r.pkg): $(r.range_lower) - $(r.range_upper)")
        end
    end

    if !isempty(stale)
        println("\nSTALE LOWER BOUNDS (should be bumped):")
        println("-"^60)
        Printf.@printf("  %-30s %15s → %-15s\n", "Package", "Current", "Minimum")
        for r in stale
            Printf.@printf("  %-30s %15s → %-15s\n", r.pkg, r.range_lower, r.min_resolvable)
        end
    end

    if !isempty(ok)
        println("\nOK (lower bound resolves):")
        println("-"^60)
        for r in ok
            println("  $(r.pkg) $(r.range_lower)")
        end
    end

    return dead, stale, ok
end

function suggested_compat(results::Vector{CompatResult})
    # Group results by package
    by_pkg = Dict{String,Vector{CompatResult}}()
    for r in results
        push!(get!(by_pkg, r.pkg, CompatResult[]), r)
    end

    suggestions = Dict{String,String}()
    for (pkg, pkg_results) in by_pkg
        surviving = String[]
        for r in pkg_results
            if r.lower_resolves
                push!(surviving, _format_bound(r.range_lower))
            elseif r.min_resolvable !== nothing
                push!(surviving, _format_bound(r.min_resolvable))
            end
            # Dead ranges are simply dropped
        end
        if !isempty(surviving)
            suggestions[pkg] = join(surviving, ", ")
        end
    end

    return suggestions
end

function _format_bound(v::VersionNumber)
    if v.patch == 0
        if v.minor == 0
            return string(v.major)
        else
            return "$(v.major).$(v.minor)"
        end
    else
        return string(v)
    end
end

function update_project_toml!(project_dir::String, suggestions::Dict{String,String})
    path = joinpath(project_dir, "Project.toml")
    lines = readlines(path)
    in_compat = false
    new_lines = String[]

    for line in lines
        if startswith(line, "[compat]")
            in_compat = true
            push!(new_lines, line)
            continue
        elseif startswith(line, "[") && in_compat
            in_compat = false
        end

        if in_compat
            m = match(r"^(\w+)\s*=\s*\"(.*)\"", line)
            if m !== nothing
                pkg = m.captures[1]
                if haskey(suggestions, pkg)
                    push!(new_lines, "$pkg = \"$(suggestions[pkg])\"")
                    continue
                end
            end
        end

        push!(new_lines, line)
    end

    open(path, "w") do io
        for (i, line) in enumerate(new_lines)
            print(io, line)
            if i < length(new_lines)
                println(io)
            end
        end
        println(io)  # trailing newline
    end
end

# ============================================================================
# Saving / loading suggestions
# ============================================================================

const SUGGESTIONS_FILE = joinpath(PROJECT_DIR, ".compat_suggestions.toml")

function save_suggestions(suggestions::Dict{String,String})
    open(SUGGESTIONS_FILE, "w") do io
        println(io, "# Auto-generated by check_lower_compat_bounds.jl")
        println(
            io,
            "# Apply with: julia .claude/skills/check-lower-compat-bounds/check_lower_compat_bounds.jl --apply\n",
        )
        println(io, "[compat]")
        for (pkg, s) in sort(collect(suggestions))
            println(io, "$pkg = \"$s\"")
        end
    end
    println("\nSuggestions saved to $SUGGESTIONS_FILE")
end

function load_suggestions()::Dict{String,String}
    if !isfile(SUGGESTIONS_FILE)
        error("No saved suggestions found at $SUGGESTIONS_FILE. Run without --apply first.")
    end
    data = TOML.parsefile(SUGGESTIONS_FILE)
    return Dict{String,String}(k => v for (k, v) in get(data, "compat", Dict()))
end

# ============================================================================
# Entry point
# ============================================================================

function print_suggestions(suggestions, results)
    println("\nSUGGESTED COMPAT UPDATES:")
    println("-"^60)
    current = Dict(r.pkg => r.current_compat for r in results)
    for (pkg, s) in sort(collect(suggestions))
        cur = get(current, pkg, "")
        if s != cur
            println("  $pkg = \"$s\"")
        end
    end
end

function main()
    if "--apply" in ARGS
        println("Applying saved suggestions to Project.toml...")
        suggestions = load_suggestions()
        update_project_toml!(PROJECT_DIR, suggestions)
        println("Done. Review the changes with `git diff Project.toml`.")
        return
    end

    println("Checking lower compat bounds in $PROJECT_DIR")
    println("This may take a while (one temp environment per test)...\n");
    flush(stdout)

    results = check_all_bounds(PROJECT_DIR)
    dead, stale, ok = print_report(results)

    if isempty(dead) && isempty(stale)
        println("\nAll compat bounds are up to date!")
        return
    end

    suggestions = suggested_compat(results)
    print_suggestions(suggestions, results)
    save_suggestions(suggestions)

    print("\nApply these updates to Project.toml? [y/N] ");
    flush(stdout)
    response = strip(readline())
    if lowercase(response) in ("y", "yes")
        update_project_toml!(PROJECT_DIR, suggestions)
        println("Done. Review the changes with `git diff Project.toml`.")
    else
        println("No changes made. Run with --apply later to apply.")
    end
end

main()
