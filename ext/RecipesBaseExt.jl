module RecipesBaseExt

using RecipesBase
using ProbNumDiffEq
using Statistics
import ProbNumDiffEq: AbstractProbODESolution
import SciMLBase: interpret_vars, getsyms


@recipe function f(
    sol::AbstractProbODESolution;
    idxs=nothing,
    denseplot=sol.dense,
    plotdensity=1000,
    tspan=nothing,
    ribbon_width=1.96,
    vars=nothing,
)
    if vars !== nothing
        Base.depwarn(
            "To maintain consistency with solution indexing, keyword argument `vars` will be removed in a future version. Please use keyword argument `idxs` instead.",
            :f; force=true)
        (idxs !== nothing) &&
            error(
                "Simultaneously using keywords `vars` and `idxs` is not supported. Please only use idxs.",
            )
        idxs = vars
    end

    tstart, tend = isnothing(tspan) ? (sol.t[1], sol.t[end]) : tspan
    times = denseplot ? range(tstart, tend, length=plotdensity) : sol.t
    sol_rvs = denseplot ? sol(times).u : sol.pu
    if !isnothing(tspan)
        sol_rvs = sol_rvs[tstart.<=times.<=tend]
        times = times[tstart.<=times.<=tend]
    end
    values = stack(mean.(sol_rvs))'
    stds = stack(std.(sol_rvs))'

    if isnothing(idxs)
        ribbon --> ribbon_width * stds
        xguide --> "t"
        yguide --> "u(t)"
        label --> hcat(["u$(i)(t)" for i in 1:length(sol.u[1])]...)
        return times, values
    else
        int_vars = interpret_vars(idxs, sol, getsyms(sol))
        varsizes = unique(length.(int_vars))
        @assert length(varsizes) == 1 "`idxs` argument has an errors"
        ndims = varsizes[1] - 1  # First argument is not about dimensions
        if ndims == 2
            _x = []
            _y = []
            _labels = []
            for (_, i, j) in int_vars
                push!(_x, i == 0 ? times : values[:, i])
                push!(_y, j == 0 ? times : values[:, j])
            end
            return _x, _y
        elseif ndims == 3
            _x = []
            _y = []
            _z = []
            for (_, i, j, k) in int_vars
                push!(_x, i == 0 ? times : values[:, i])
                push!(_y, j == 0 ? times : values[:, j])
                push!(_z, k == 0 ? times : values[:, k])
            end
            return _x, _y, _z
        else
            error("Error with `idxs` argument")
        end
    end
end

end
