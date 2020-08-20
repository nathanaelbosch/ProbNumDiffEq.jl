# Goal: Test the correctness of state initialization


a, b = 1.1, -0.5
f(u, p, t) = [a*u[1], b*u[2]]
u0 = [0.1, 1.0]
tspan = (0.0, 5.0)
t0, T = tspan
prob = ODEProblem(f, u0, tspan)
p = prob.p

# True Solutions and derivatives
u(t) = [a^0*u0[1] * exp(a*t), u0[2] * exp(b*t)]
du(t) = [a^1*u0[1] * exp(a*t), b * u0[2] * exp(b*t)]
ddu(t) = [a^2*u0[1] * exp(a*t), (b)^2 * u0[2] * exp(b*t)]
dddu(t) = [a^3*u0[1] * exp(a*t), (b)^3 * u0[2] * exp(b*t)]
ddddu(t) = [a^4*u0[1] * exp(a*t), (b)^4 * u0[2] * exp(b*t)]
dddddu(t) = [a^5*u0[1] * exp(a*t), (b)^5 * u0[2] * exp(b*t)]
ddddddu(t) = [a^6*u0[1] * exp(a*t), (b)^6 * u0[2] * exp(b*t)]
true_init_states = [u(t0); du(t0); ddu(t0); dddu(t0); ddddu(t0); dddddu(t0); ddddddu(t0)]


@testset "OOP state init" begin
    m0, P0 = ProbNumODE.get_initial_states_forwarddiff(prob, 6)
    @test m0 ≈ true_init_states
    @test all(P0 .== 0)
end


@testset "IIP state init" begin
    f!(du, u, p, t) = (du .= f(u, p, t))
    prob = ODEProblem(f!, u0, tspan)
    m0, P0 = ProbNumODE.get_initial_states_forwarddiff(prob, 6)
    @test m0 ≈ true_init_states
    @test all(P0 .== 0)
end
