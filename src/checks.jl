function check_secondorderode(integ)
    if integ.f isa DynamicalODEFunction &&
       !(integ.sol.prob.problem_type isa SecondOrderODEProblem)
        error(
            """
          The given problem is a `DynamicalODEProblem`, but not a `SecondOrderODEProblem`.
          This can not be handled by ProbNumDiffEq.jl right now. Please check if the
          problem can be formulated as a second order ODE. If not, please open a new
          github issue!
          """,
        )
    end
end
function check_densesmooth(integ)
    if integ.opts.dense && !integ.alg.smooth
        error("To use `dense=true` you need to set `smooth=true`!")
    end
    if !integ.opts.save_everystep && integ.alg.smooth
        error("If you set `save_everystep=false` also set `smooth=false` in the alg!")
    end
    if !isempty(integ.opts.saveat)
        error(
            "`saveat` is not supported. " *
            "Solve with `save_everystep=true` and index the solution at the desired times instead.",
        )
    end
end
function check_saveiter(integ)
    @assert integ.saveiter == 1
end
"""
    check_forward_in_time(integ)

Throw an `ArgumentError` if the time span runs backward in time, which is not supported.
"""
function check_forward_in_time(integ)
    t0, tend = integ.sol.prob.tspan
    if tend < t0
        throw(
            ArgumentError(
                "The time span $((t0, tend)) runs backward in time. " *
                "ProbNumDiffEq.jl only supports integration forward in time.",
            ),
        )
    end
end
"""
    check_nonnegative_dt(dt)

Throw an `ArgumentError` for a negative (initial) step size, i.e. a solve backward in time.
Needed in addition to `check_forward_in_time`, since fixed-step solves with a negative `dt`
already fail while building the cache.
"""
function check_nonnegative_dt(dt)
    if dt < zero(dt)
        throw(
            ArgumentError(
                "Negative step sizes (here `dt = $dt`) are not supported. " *
                "ProbNumDiffEq.jl only supports integration forward in time.",
            ),
        )
    end
end
