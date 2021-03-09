########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution;
                   vars=nothing,
                   denseplot=sol.dense,
                   plotdensity=1000,
                   ribbon_width=1.96)

    times = denseplot ? range(sol.t[1], sol.t[end], length=plotdensity) : sol.t
    sol_rvs = denseplot ? sol(times).u : sol.pu
    values = stack(mean(sol_rvs))
    stds = stack(std(sol_rvs))

    if isnothing(vars)
        ribbon --> ribbon_width * stds
        xguide --> "t"
        yguide --> "u(t)"
        label --> hcat(["u$(i)(t)" for i in 1:length(sol.u[1])]...)
        return times, values
    else
        _times = []
        _values = []
        for (_, i, j) in DiffEqBase.interpret_vars(vars, sol, DiffEqBase.getsyms(sol))
            push!(_times, i == 0 ? times : values[:, i])
            push!(_values, j == 0 ? times : values[:, j])
        end
        return _times, _values
    end
end
