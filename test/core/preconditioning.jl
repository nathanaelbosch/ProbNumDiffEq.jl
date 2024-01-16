using Test
using LinearAlgebra
using ProbNumDiffEq
import ProbNumDiffEq as PNDE
import ODEProblemLibrary: prob_ode_lotkavolterra

prob = prob_ode_lotkavolterra
prob = remake(prob, tspan=(0.0, 10.0))

@testset "Condition numbers of A,Q" begin
    h = 0.1 * rand()
    σ = rand()

    d, q = 2, 3

    prior = PNDE.IWP(dim=d, num_derivatives=q)

    Ah, Qh = PNDE.discretize(prior, h)
    Qh = PNDE.apply_diffusion(Qh, σ^2)

    Ah_p, Qh_p = PNDE.preconditioned_discretize(prior)
    Qh_p = PNDE.apply_diffusion(Qh_p, σ^2)

    # First test that they're both equivalent
    D = d * (q + 1)
    P, PI = PNDE.init_preconditioner(PNDE.DenseCovariance{Float64}(d, q))
    PNDE.make_preconditioner!(P, h, d, q)
    PNDE.make_preconditioner_inv!(PI, h, d, q)
    @test Ah_p ≈ P * Ah * PI
    @test Matrix(Qh_p) ≈ P * Matrix(Qh) * P'

    # Check that the preconditioning actually helps
    # @info "Condition numbers" cond(Qh) cond(Matrix(Qh_p)) cond(Ah) cond(Ah_p)
    @test cond(Matrix(Qh)) > cond(Matrix(Qh_p))
    @test cond(Matrix(Qh)) > cond(Matrix(Qh_p))^2
end
