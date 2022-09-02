using ProbNumDiffEq
using Test
import ODEProblemLibrary: prob_ode_linear

@testset "$alg" for alg in (EK0(), EK1())
    @test_broken solve(prob_ode_linear, alg) isa ProbNumDiffEq.ProbODESolution
end
