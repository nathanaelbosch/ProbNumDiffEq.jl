using Test
using LinearAlgebra
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems;
importodeproblems();
import DiffEqProblemLibrary.ODEProblemLibrary:
    prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo

prob = prob_ode_lotkavoltera
prob = remake(prob, tspan=(0.0, 10.0))
prob = ProbNumDiffEq.remake_prob_with_jac(prob)

@testset "Condition numbers of A,Q" begin
    h = 0.1 * rand()
    σ = rand()

    d, q = 2, 3

    A!, Q! = ProbNumDiffEq.vanilla_ibm(d, q)
    Ah = diagm(0 => ones(d * (q + 1)))
    Qh = zeros(d * (q + 1), d * (q + 1))
    A!(Ah, h)
    Q!(Qh, h, σ^2)

    A_p, Q_p = ProbNumDiffEq.ibm(d, q)
    Ah_p = A_p
    Qh_p = Q_p * σ^2

    # First test that they're both equivalent
    D = d * (q + 1)
    P, PI = ProbNumDiffEq.init_preconditioner(d, q)
    ProbNumDiffEq.make_preconditioner!(P, h, d, q)
    ProbNumDiffEq.make_preconditioner_inv!(PI, h, d, q)
    @test Qh_p ≈ P * Qh * P'
    @test Ah_p ≈ P * Ah * PI

    # Check that the preconditioning actually helps
    # @info "Condition numbers" cond(Qh) cond(Matrix(Qh_p)) cond(Ah) cond(Ah_p)
    @test cond(Qh) > cond(Matrix(Qh_p))
    @test cond(Qh) > cond(Matrix(Qh_p))^2
end
