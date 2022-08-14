# Solving DAEs with Probabilistic Numerics

ProbNumDiffEq.jl provides probabilistic numerical solvers for _differential algebraic equations_ (DAEs).
Currently, we recommend using the semi-implicit `EK1` algorithm.

!!! note
    
    For a more general tutorial on DAEs, solved with classic solvers, check out the
    [DifferentialEquations.jl DAE tutorial](https://diffeq.sciml.ai/stable/tutorials/dae_example/).

#### Solving a Mass-Matrix DAE with the `EK1`

```@example 2
using ProbNumDiffEq, Plots

function rober(du, u, p, t)
    y₁, y₂, y₃ = u
    k₁, k₂, k₃ = p
    du[1] = -k₁ * y₁ + k₃ * y₂ * y₃
    du[2] = k₁ * y₁ - k₃ * y₂ * y₃ - k₂ * y₂^2
    du[3] = y₁ + y₂ + y₃ - 1
    nothing
end
M = [1.0 0 0
    0 1.0 0
    0 0 0]
f = ODEFunction(rober, mass_matrix=M)
prob_mm = ODEProblem(f, [1.0, 0.0, 0.0], (0.0, 1e5), (0.04, 3e7, 1e4))
using Logging;
Logging.disable_logging(Logging.Warn); # hide
sol = solve(prob_mm, EK1(), reltol=1e-8, abstol=1e-8)
Logging.disable_logging(Logging.Debug) # hide
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

### References

[1] N. Bosch, F. Tronarp, P. Hennig: **Pick-and-Mix Information Operators for Probabilistic ODE Solvers** (2022) ([link](https://arxiv.org/abs/2110.10770))
