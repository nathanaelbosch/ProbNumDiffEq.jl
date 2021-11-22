# Copied from DiffEqDevTools.jl and modified
# See also: https://devdocs.sciml.ai/dev/alg_dev/test_problems/
function DiffEqDevTools.appxtrue(sol::ProbODESolution, sol2::TestSolution)

    # (almost) everything should work on the mean solution
    sol = mean(sol)

    if sol2.u == nothing && DiffEqDevTools.hasinterp(sol2)
        _sol = TestSolution(sol.t, sol2(sol.t), sol2)
    else
        _sol = sol2
    end

    errors = Dict(:final => DiffEqDevTools.recursive_mean(abs.(sol[end] - _sol[end])))
    if _sol.dense
        timeseries_analytic = _sol(sol.t)
        ts_err = sol - timeseries_analytic
        errors[:l∞] = maximum(DiffEqDevTools.vecvecapply((x) -> abs.(x), ts_err))
        errors[:l2] = sqrt(
            DiffEqDevTools.recursive_mean(
                DiffEqDevTools.vecvecapply((x) -> float(x) .^ 2, ts_err),
            ),
        )

        if sol.dense
            densetimes = collect(range(sol.t[1], stop=sol.t[end], length=100))
            interp_u = sol(densetimes)
            interp_analytic = _sol(densetimes)
            interp_err = interp_u - interp_analytic
            interp_errors = Dict(
                :L∞ => maximum(DiffEqDevTools.vecvecapply((x) -> abs.(x), interp_err)),
                :L2 => sqrt(
                    DiffEqDevTools.recursive_mean(
                        DiffEqDevTools.vecvecapply((x) -> float(x) .^ 2, interp_err),
                    ),
                ),
            )
            errors = merge(errors, interp_errors)
        end
    else
        timeseries_analytic = sol2.u
        if sol.t == sol2.t
            ts_err = sol - timeseries_analytic
            errors[:l∞] = maximum(vecvecapply((x) -> abs.(x), ts_err))
            errors[:l2] = sqrt(recursive_mean(vecvecapply((x) -> float(x) .^ 2, ts_err)))
        end
    end
    return DiffEqBase.build_solution(sol, timeseries_analytic, errors)
end
