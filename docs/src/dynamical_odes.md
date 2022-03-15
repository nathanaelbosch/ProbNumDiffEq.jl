# Second Order ODEs and Energy Preservation
In this tutorial we consider an _energy-preserving_, physical dynamical system, given by a _second-order_ ODE.


#### TL;DR:
1. To _efficiently_ solve second-order ODEs, just define the problem as a `SecondOrderODEProblem`.
2. To preserve constant quantities, use the `ManifoldUpdate` callback; same syntax as
   [DiffEqCallback.jl's `ManifoldProjection`](https://diffeq.sciml.ai/stable/features/callback_library/#Manifold-Conservation-and-Projection).


## Simulating the Hénon-Heiles system
The Hénon-Heiles model describes the motion of a star around a galactic center, restricted to a plane.
It is given by a second-order ODE
```math
\begin{aligned}
\ddot{x} &= - x - 2 x y \\
\ddot{y} &= y^2 - y - x^2.
\end{aligned}
```
Our goal is to numerically simulate this system on a time span
``t \in [0, T]``, starting with initial values
``x(0)=0``, ``y(0) = 0.1``, ``\dot{x}(0) = 0.5``, ``\dot{y}(0) = 0``.


### Transforming the problem into a first-order ODE
A very common approach is to first transform the problem into a first-order ODE by introducing a new variable
```math
u = [dx,dy,x,y],
```
to obtain
```math
\begin{aligned}
\dot{u}_1(t) &= - u_3 - 2 u_3 u_4 \\
\dot{u}_2(t) &= u_4^2 - u_4 - u_4^2 \\
\dot{u}_3(t) &= u_1 \\
\dot{u}_4(t) &= u_2.
\end{aligned}
```

This first-order ODE can then be solved using any conventional ODE solver - including our `EK1`:

```@example 2
using ProbNumDiffEq, Plots

function Hénon_Heiles(du,u,p,t)
    du[1] = -u[3] - 2*u[3]*u[4]
    du[2] = u[4]^2 - u[4] -u[3]^2
    du[3] = u[1]
    du[4] = u[2]
end
u0, du0 = [0.0, 0.1], [0.5, 0.0]
tspan = (0.0, 100.0)
prob = ODEProblem(Hénon_Heiles, [du0; u0], tspan)
sol = solve(prob, EK1());
plot(sol, vars=(3,4)) # where `vars=(3,4)` is used to plot x agains y
```

### Solving the second-order ODE directly
Instead of first transforming the problem, we can also solve it directly as a second-order ODE, by defining it as a `SecondOrderODEProblem`.

!!! note
    The `SecondOrderODEProblem` type is not defined in ProbNumDiffEq.jl but is provided by SciMLBase.jl.
    For more information, check out the DifferentialEquations.jl documentation on [Dynamical, Hamiltonian and 2nd Order ODE Problems](https://diffeq.sciml.ai/stable/types/dynamical_types/).

```@example 2
function Hénon_Heiles2(ddu,du,u,p,t)
    ddu[1] = -u[1] - 2*u[1]*u[2]
    ddu[2] = u[2]^2 - u[2] -u[1]^2
end
prob2 = SecondOrderODEProblem(Hénon_Heiles2, du0, u0, tspan)
sol2 = solve(prob2, EK1());
plot(sol2, vars=(3,4))
```

### Benchmark: Solving second order ODEs is _faster_
Solving second-order ODEs is not just a matter of convenience - in fact, SciMLBase's `SecondOrderODEProblem` is neatly designed in such a way that all the classic solvers from OrdinaryDiffEq.jl can handle it by solving the corresponding first-order ODE.
But, transforming the ODE to first order increases the dimensionality of the problem, and comes therefore at increased computational cost; this also motivates [classic specialized solvers for second-order ODEs](https://diffeq.sciml.ai/stable/solvers/dynamical_solve/).

The probablistic numerical solvers from ProbNumDiffEq.jl have the same internal state representation for first and second order ODEs; all that changes is the _measurement model_ [1].
As a result, we can use the `EK1` both for first and second order ODEs, but it automatically specializes on the latter to provide a __2x performance boost__:
```
julia> @btime solve(prob, EK1(order=3), adaptive=false, dt=1e-2);
  766.312 ms (400362 allocations: 173.38 MiB)

julia> @btime solve(prob2, EK1(order=4), adaptive=false, dt=1e-2);
  388.301 ms (510676 allocations: 102.78 MiB)
```


## Energy preservation
In addition to the ODE given above, we know that the solution of the Hénon-Heiles model has to _preserve energy_ over time.
The total energy can be expressed as the sum of the potential and kinetic energies, given by
```math
\begin{aligned}
\operatorname{PotentialEnergy}(x,y) &= \frac{1}{2} \left( x^2 + y^2 + 2 x^2 y - \frac{2y^3}{3} \right), \\
\operatorname{KineticEnergy}(\dot{x}, \dot{y}) &= \frac{1}{2} \left( \dot{x}^2 + \dot{y}^2 \right).
\end{aligned}
```

In code:
```@example 2
PotentialEnergy(x,y) = 1//2 * (x^2 + y^2 + 2x^2*y - 2//3 * y^3)
KineticEnergy(dx,dy) = 1//2 * (dx^2 + dy^2)
E(dx,dy,x,y) = PotentialEnergy(x,y) + KineticEnergy(dx,dy)
E(u) = E(u...); # convenient shorthand
```

So, let's have a look at how the total energy changes over time when we numerically simulate the Hénon-Heiles model over a long period of time:
Standard solve
```@example 2
longprob = remake(prob2, tspan=(0.0, 1e3))
longsol = solve(longprob, EK1(smooth=false), dense=false)
plot(longsol.t, E.(longsol.u))
```
It visibly loses energy over time, from an initial 0.12967 to a final 0.12899.
Let's fix this to get a physically more meaningful solution.

### Energy preservation with the `ManifoldUpdate` callback
In the language of ODE filters, preserving energy over time amounts to just another measurement model [1].
The most convenient way of updating on this additional zero measurement with ProbNumDiffEq.jl is with the `ManifoldUpdate` callback.

!!! note
    The `ManifoldUpdate` callback can be thought of a probabilistic counterpart to the [`ManifoldProjection`](https://diffeq.sciml.ai/stable/features/callback_library/#Manifold-Conservation-and-Projection) callback provided by DiffEqCallbacks.jl.

To do so, first define a (vector-valued) residual function, here chosen to be the difference between the current energy and the initial energy, and build a `ManifoldUpdate` callback
```@example 2
residual(u) = [E(u) - E(du0..., u0...)]
cb = ManifoldUpdate(residual)
```

Then, solve the ODE with this callback
```@example 2
longsol_preserving = solve(longprob, EK1(smooth=false), dense=false, callback=cb)
plot(longsol.t, E.(longsol.u))
plot!(longsol_preserving.t, E.(longsol_preserving.u))
```

Voilà! With the `ManifoldUpdate` callback we could preserve the energy over time and obtain a more truthful probabilistic numerical long-term simulation of the Hénon-Heiles model.


#### References
[1] N. Bosch, F. Tronarp, P. Hennig: **Pick-and-Mix Information Operators for Probabilistic ODE Solvers** (2022)
