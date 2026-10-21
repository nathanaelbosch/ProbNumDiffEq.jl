using Test
using ProbNumDiffEq
using LinearAlgebra
using ForwardDiff
import ProbNumDiffEq as PNDE

function f!(du, u, p, t)
    du[1] = p * (u[1] - u[1]^3 / 3 + u[2]) + sin(t)
    du[2] = -(u[1] - 0.2 + 0.2u[2]) / p
    return nothing
end
f(u, p, t) = (du=similar(u); f!(du, u, p, t); du)
M = Diagonal([1.0, 2.0])
prob = ODEProblem(ODEFunction(f!; mass_matrix=M), [-1.0, 1.0], (0.0, 1.0), 3.0)
d, q = 2, 3
E0, E1 = PNDE.projection(d, q)(0), PNDE.projection(d, q)(1)

@testset "$S linearizes at a point other than the predicted mean" for (S, approx) in (
    EK0 => J -> zero(J),
    EK1 => identity,
    DiagonalEK1 => J -> Diagonal(J),
)
    integ = init(prob, S(order=q))
    t = integ.t + integ.dt
    m = collect(range(0.5, 1.5, length=d * (q + 1)))
    @assert m != integ.cache.x_pred.μ

    obs = PNDE.linearize_ode_observation!(integ.alg.linearization, integ, m)
    @test obs.m === m
    @test obs.r ≈ M * E1 * m - f(E0 * m, prob.p, t)
    J = ForwardDiff.jacobian(u -> f(u, prob.p, t), E0 * m)
    @test Matrix(obs.H) ≈ M * E1 - approx(J) * E0

    x = integ.cache.x_pred
    @test PNDE.linearize_ode_observation!(integ.alg.linearization, integ, x).m === x.μ
end

@testset "Residual and observation operator of the MatfreeEK1 at a point" begin
    jacobian(u, p, t) = ForwardDiff.jacobian(u -> f(u, p, t), u)
    jvp!(Jv, v, u, p, t) = mul!(Jv, jacobian(u, p, t), v)
    vjp!(Jtv, v, u, p, t) = mul!(Jtv, jacobian(u, p, t)', v)
    prob = ODEProblem(ODEFunction(f!; jvp=jvp!, vjp=vjp!), [-1.0, 1.0], (0.0, 1.0), 3.0)
    integ = init(prob, MatfreeEK1(order=q))
    t = integ.t + integ.dt
    m = collect(range(0.5, 1.5, length=d * (q + 1)))

    obs = PNDE.linearize_ode_observation!(integ.alg.linearization, integ, m)
    @test obs.m === m
    @test obs.r ≈ E1 * m - f(E0 * m, prob.p, t)
    J = jacobian(E0 * m, prob.p, t)
    x = collect(range(-1.0, 1.0, length=d * (q + 1)))
    @test mul!(zeros(d), obs.H, x) ≈ (E1 - J * E0) * x
end
