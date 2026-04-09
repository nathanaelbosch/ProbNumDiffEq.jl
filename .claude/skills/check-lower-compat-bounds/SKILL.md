---
name: check-lower-compat-bounds
description: Test whether the lower compat bounds in Project.toml are still resolvable. Identifies stale bounds, dead version ranges, and suggests updates.
---

You are checking the lower compat bounds of a Julia package for staleness.

## How it works

The script `check_lower_compat_bounds.jl` (in the same directory as this skill) does the heavy lifting:
- Parses all compat entries from Project.toml
- For each dependency, tests whether the lower bound resolves in a fresh temp environment
- Binary searches for the true minimum resolvable version when a bound fails
- Reports dead ranges, stale bounds, and OK bounds
- Saves suggestions to `.compat_suggestions.toml` and prompts to apply

## Steps

1. Run the script and let it complete (it takes a while, one temp environment per test):

```
julia .claude/skills/check-lower-compat-bounds/check_lower_compat_bounds.jl
```

2. The script will print a report and prompt to apply changes. If the user declines, the suggestions are saved and can be applied later with:

```
julia .claude/skills/check-lower-compat-bounds/check_lower_compat_bounds.jl --apply
```

3. Do NOT commit automatically. Let the user decide how to handle that.

## Important context

- A bound that fails resolution is definitively stale: no user can install that version combination. This is caused by transitive dependency constraints, not bugs in this package.
- A bound that passes resolution is not guaranteed to pass tests. If the user wants deeper validation, offer to run the test suite with specific deps pinned to their lower bounds.
- The script skips stdlibs (LinearAlgebra, Printf, etc.) since those are tied to the Julia version.
