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
        @assert length(unique(length.(vars)))==1 "`vars` argument has an errors"
        ndims = vars isa Tuple && all(length.(vars) .== 1) ?
            length(vars) : unique(length.(vars))[1]
        @info "?" vars ndims
        if ndims ==  2
            _x = []
            _y = []
            _labels = []
            int_vars = DiffEqBase.interpret_vars(vars, sol, DiffEqBase.getsyms(sol))
            for (_, i, j) in int_vars
                push!(_x, i == 0 ? times : values[:, i])
                push!(_y, j == 0 ? times : values[:, j])
            end
            return _x, _y
        elseif ndims == 3
            _x = []
            _y = []
            _z = []
            int_vars = DiffEqBase.interpret_vars(vars, sol, DiffEqBase.getsyms(sol))
            for (_, i, j, k) in int_vars
                push!(_x, i == 0 ? times : values[:, i])
                push!(_y, j == 0 ? times : values[:, j])
                push!(_z, k == 0 ? times : values[:, k])
            end
            return _x, _y, _z
        end
    end
end
