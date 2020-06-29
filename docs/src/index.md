# ProbNumODE.jl Documentation


## Minimal Example
```julia
using ProbNumODE

prob = fitzhugh_nagumo()
sol = solve(prob, ODEFilter())
```


## Plotting Functionality
Requires [Plots.jl](https://github.com/JuliaPlots/Plots.jl).
```julia
using Plots
plot(sol)
```
![Fitzhugh-Nagumo Solution](./figures/fitzhugh_nagumo.png?raw=true "Fitzhugh-Nagumo Solution")


## Compatible with problems from DiffEqProblemLibrary.jl
Requires [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl)
```julia
using DiffEqProblemLibrary
DiffEqProblemLibrary.ODEProblemLibrary.importodeproblems()

prob = DiffEqProblemLibrary.ODEProblemLibrary.prob_ode_rigidbody
sol = solve(prob, ODEFilter())
plot(sol)
```
![Rigid Body Equations](./figures/rigidbody.png?raw=true "Rigid Body Equations Solution")


## Symbolic problem definition
Requires [DifferentialEquations.jl](https://docs.sciml.ai/stable/)
```julia
using DifferentialEquations
using ProbNumODE

f = @ode_def begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d
p = [1.5,1.0,3.0,1.0]
u0 = [1.0,1.0]
tspan = (0.0,10.0)
prob = ODEProblem(f,u0,tspan,p)
sol = solve(prob, ODEFilter())
plot(sol)
```
![Lotka-Volterra Solution](./figures/lotka-volterra.png?raw=true "Lotka-Volterra Solution")
