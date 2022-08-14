#### Fun-Fact: Solving an Index-3 DAE

The following is based on the
["Automatic Index Reduction of DAEs"](https://mtk.sciml.ai/stable/mtkitize_tutorials/modelingtoolkitize_index_reduction/)
tutorial by
[ModelingToolkit.jl](https://mtk.sciml.ai/stable/), which demonstrates how the classic `Rodas4` solver fails to solve a DAE due to the fact that it is of index 3; which is why ModelingToolkit's automatic index reduction is so useful.
It turns out that __our probabilistic numerical solvers can directly solve the index-3 DAE__:

```@example 2
using ProbNumDiffEq, LinearAlgebra
function pendulum!(du, u, p, t)
    x, dx, y, dy, T = u
    g, L = p
    du[1] = dx
    du[2] = T * x
    du[3] = dy
    du[4] = T * y - g
    du[5] = x^2 + y^2 - L^2
end
pendulum_fun! = ODEFunction(pendulum!, mass_matrix=Diagonal([1, 1, 1, 1, 0]))
u0 = [1.0, 0, 0, 0, 0];
p = [9.8, 1];
tspan = (0, 10.0);
pendulum_prob = ODEProblem(pendulum_fun!, u0, tspan, p)
sol = solve(pendulum_prob, EK1())
plot(sol)
```

```@example 2
prob_index3 = ODEProblem(modelingtoolkitize(pendulum_prob), Pair[], tspan)

traced_sys = modelingtoolkitize(pendulum_prob)
pendulum_sys = structural_simplify(dae_index_lowering(traced_sys))
prob_index1 = ODEProblem(pendulum_sys, Pair[], tspan)

traced_sys = modelingtoolkitize(pendulum_prob)
pendulum_sys = structural_simplify(dae_index_lowering(traced_sys))
prob_odae = ODAEProblem(pendulum_sys, Pair[], tspan)
```
