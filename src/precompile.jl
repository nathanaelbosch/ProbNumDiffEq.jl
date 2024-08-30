# Inspired by:
# https://github.com/SciML/OrdinaryDiffEq.jl/blob/v6.21.0/src/OrdinaryDiffEqCore.jl#L195-L221

import PrecompileTools

PrecompileTools.@compile_workload begin
    function lorenz(du, u, p, t)
        du[1] = 10.0(u[2] - u[1])
        du[2] = u[1] * (28.0 - u[3]) - u[2]
        du[3] = u[1] * u[2] - (8 / 3) * u[3]
        return nothing
    end

    prob_list = [
        ODEProblem{true,true}(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0)),
        ODEProblem{true,false}(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0)),
        ODEProblem{true,false}(lorenz, [1.0; 0.0; 0.0], (0.0, 1.0), Float64[]),
    ]

    solver_list = [
        EK0()
        EK1()
    ]

    for prob in prob_list, solver in solver_list
        solve(prob, solver)(5.0)
    end
end
