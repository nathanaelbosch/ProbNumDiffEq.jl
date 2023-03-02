# Solving DAEs with Probabilistic Numerics

ProbNumDiffEq.jl provides probabilistic numerical solvers for _differential algebraic equations_ (DAEs).
Currently, we recommend using the semi-implicit `EK1` algorithm.

!!! note

    For a more general tutorial on DAEs check out the
    [DifferentialEquations.jl DAE tutorial](https://diffeq.sciml.ai/stable/tutorials/dae_example/).

## Solving mass-matrix DAEs with the `EK1`

First, define the DAE (here the ROBER problem) as an ODE problem with singular mass matrix:
```@example 2
using ProbNumDiffEq, Plots, LinearAlgebra, OrdinaryDiffEq, ModelingToolkit

function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    nothing
end
M = [1 0 0
     0 1 0
     0 0 0]
f = ODEFunction(rober, mass_matrix=M)
prob_mm = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e5), (0.04, 3e7, 1e4))
```

We can solve this problem directly with the `EK1`:
```@example 2
sol = solve(prob_mm, EK1(), reltol=1e-8, abstol=1e-8)
plot(
    sol,
    xscale=:log10,
    tspan=(1e-6, 1e5),
    layout=(3, 1),
    legend=false,
    ylabel=["u₁(t)" "u₂(t)" "u₃(t)"],
    xlabel=["" "" "t"],
    denseplot=false,
)
```
Looks good!



## Solving an Index-3 DAE directly

The following is based on the
["Automatic Index Reduction of DAEs"](https://docs.sciml.ai/ModelingToolkit/stable/examples/modelingtoolkitize_index_reduction/)
tutorial by
[ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/),
which demonstrates how the classic `Rodas4` solver fails to solve a DAE due to the fact that it is of index 3; which is why ModelingToolkit's automatic index reduction is so useful.

__It turns out that our probabilistic numerical solvers can directly solve the index-3 DAE!__

First, define the pendulum problem as in the tutorial:
```@example 2
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
tspan = (0, 5.0);
pendulum_prob = ODEProblem(pendulum_fun!, u0, tspan, p)
```

We can try to solve it directly with one of the classic mass-matrix DAE solvers from OrdinaryDiffEq.jl:
```@example 2
solve(pendulum_prob, Rodas4())
```

It does not work!
This is because of the _index_ of the DAE; see e.g. [this explenation from the tutorial](https://docs.sciml.ai/ModelingToolkit/stable/examples/modelingtoolkitize_index_reduction/#Understanding-DAE-Index).

Does this also hold for the `EK1` solver? Let's find out:
```@example 2
sol = solve(pendulum_prob, EK1())
```
Nope! The `EK1` is able to solve the index-3 DAE directly. Pretty cool!

```@example 2
plot(sol)
```


### Is index-reduction still worth it?

The point of the
["Automatic Index Reduction of DAEs"](https://docs.sciml.ai/ModelingToolkit/stable/examples/modelingtoolkitize_index_reduction/)
tutorial is to demonstrate ModelingToolkit's utility for automatic index reduction, which enables the classic implicit Runge-Kutta solvers such as `Rodas5` to solve this DAE.
Let's see if that still helps in this context here.

First, `modelingtoolkitize` the problem:
```@example 2
traced_sys = modelingtoolkitize(pendulum_prob)
```
(how cool is this latex output ?!?)

Next, lower the DAE index and simplify it with MTK's `dae_index_lowering` and `structural_simplify`:

```@example 2
simplified_sys = structural_simplify(dae_index_lowering(traced_sys))
```

Let's build two different ODE problems, and check how well we can solve each:
```@example 2
prob_index3 = ODEProblem(traced_sys, Pair[], tspan)
prob_index1 = ODEProblem(simplified_sys, Pair[], tspan)

sol3 = solve(prob_index3, EK1())
sol1 = solve(prob_index1, EK1())

truesol = solve(prob_index1, Rodas4(), abstol=1e-10, reltol=1e-10)

sol1_final_error = norm(sol1.u[end] - truesol.u[end])
sol1_f_evals     = sol1.destats.nf
sol3_final_error = norm(sol3.u[end] - truesol.u[end])
sol3_f_evals     = sol3.destats.nf
@info "Results" sol1_final_error sol1_f_evals sol3_final_error sol3_f_evals
```

The error for the index-1 DAE solve is _much_ lower.
So it seems that, even if the index-3 DAE could also be solved directly, index lowering might still be beneficial when solving DAEs with the `EK1`!


### References

[1] N. Bosch, F. Tronarp, P. Hennig: **Pick-and-Mix Information Operators for Probabilistic ODE Solvers** (2022) ([link](https://proceedings.mlr.press/v151/bosch22a.html))
