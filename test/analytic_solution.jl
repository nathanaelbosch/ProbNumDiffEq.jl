using ProbNumDiffEq
using Test

linear(u, p, t) = p .* u
linear_analytic(u0, p, t) = @. u0 * exp(p * t)
f = ODEFunction(linear, analytic=linear_analytic)
prob = ODEProblem(f, [1 / 2], (0.0, 1.0), 1.01)

sol = solve(prob, EK0())
@test sol.errors isa Dict
@test all(haskey.(Ref(sol.errors), (:l∞, :l2, :final)))
