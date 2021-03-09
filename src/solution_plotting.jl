########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution;
                   denseplot = sol.dense,
                   plotdensity = 1000,
                   ribbon_width=1.96)
    xguide --> "t"
    yguide --> "u(t)"
    label --> hcat(["u$(i)(t)" for i in 1:length(sol.u[1])]...)
    if denseplot
        times = range(sol.t[1], sol.t[end], length=plotdensity)
        dense_post = sol(times).u
        values = stack(mean(dense_post))
        stds = stack(std(dense_post))
        ribbon --> ribbon_width * stds
        return times, values
    else
        ribbon --> ribbon_width * stack(std.(sol.pu))
        return sol.t, stack(sol.u)
    end
end
