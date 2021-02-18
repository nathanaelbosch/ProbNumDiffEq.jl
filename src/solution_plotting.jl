########################################################################################
# Plotting
########################################################################################
@recipe function f(sol::AbstractProbODESolution; ribbon_width=1.96)
    times = range(sol.t[1], sol.t[end], length=1000)
    dense_post = sol(times).u
    values = stack(mean(dense_post))
    stds = stack(std(dense_post))
    ribbon --> ribbon_width * stds
    xguide --> "t"
    yguide --> "u(t)"
    label --> hcat(["u$(i)(t)" for i in 1:length(sol.u[1])]...)
    return times, values
end
