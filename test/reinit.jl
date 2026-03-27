using Test
using ProbNumDiffEq
using OrdinaryDiffEq

@testset "reinit!" begin
    prob = ODEProblem((du, u, p, t) -> (du .= -u), ones(2), (0.0, 1.0))

    @testset "init -> step! -> reinit! -> step! ($Alg)" for Alg in (EK0, EK1)
        integ = init(prob, Alg(order=3, smooth=false); adaptive=false, dt=0.1, dense=false)
        step!(integ)
        reinit!(integ)
        step!(integ)
        @test all(isfinite, integ.sol.diffusions[1])
    end
end
