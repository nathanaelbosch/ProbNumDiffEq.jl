## Compatible with problems from DiffEqProblemLibrary.jl
Requires [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl)
```@example
using ProbNumODE
using Plots
using DiffEqProblemLibrary
DiffEqProblemLibrary.ODEProblemLibrary.importodeproblems()

prob = DiffEqProblemLibrary.ODEProblemLibrary.prob_ode_rigidbody
sol = solve(prob, EKF1())
plot(sol)
savefig("./figures/rigidbody.svg"); nothing # hide
```
![](./figures/rigidbody.svg)


## Symbolic problem definition
Requires [DifferentialEquations.jl](https://docs.sciml.ai/stable/)
```@example
using DifferentialEquations
using ProbNumODE
using Plots

f = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d
p = [1.5,1.0,3.0,1.0]
u0 = [1.0,1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob, EKF1())
plot(sol)
savefig("./figures/lotka-volterra.svg"); nothing # hide
```
![](./figures/lotka-volterra.svg)
