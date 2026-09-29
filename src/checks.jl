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
"""
    check_local_diagonal_diffusion(integ)

Throw an `ArgumentError` if the diffusion model needs `local_diagonal_diffusion` but `H`
is not block-diagonal, `E1` or `E2`.
"""
function check_local_diagonal_diffusion(integ)
    @unpack diffusionmodel, covariance_factorization = integ.cache
    # Mirrors when `perform_step!` calls `estimate_local_diffusion`, and which diffusion
    # models then use `local_diagonal_diffusion`
    uses_local_diagonal_diffusion =
        (integ.opts.adaptive || isdynamic(diffusionmodel)) && (
            diffusionmodel isa DynamicMVDiffusion ||
            (diffusionmodel isa FixedMVDiffusion && integ.alg isa EK0)
        )
    # `local_diagonal_diffusion` needs `H` to be block-diagonal, or `E1` / `E2`
    supports_local_diagonal_diffusion =
        covariance_factorization isa BlockDiagonalCovariance ||
        (integ.alg isa EK0 && integ.f.mass_matrix == I)
    if uses_local_diagonal_diffusion && !supports_local_diagonal_diffusion
        throw(
            ArgumentError(
                "The local diagonal diffusion estimate of `$(nameof(typeof(diffusionmodel)))` " *
                "requires either the `BlockDiagonalCovariance` factorization, or the `EK0` " *
                "without a mass matrix. Use `BlockDiagonalCovariance` (the default for " *
                "`DiagonalEK1`, and for `EK0` with an `IWP` prior), or a scalar diffusion " *
                "model like `DynamicDiffusion`.",
            ),
        )
    end
end
