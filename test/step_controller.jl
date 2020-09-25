using ProbNumODE
using UnPack

using Test
using OrdinaryDiffEq
using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_linear, prob_ode_2Dlinear, prob_ode_lotkavoltera, prob_ode_fitzhughnagumo


@testset "Proportional Control" begin
    prob = prob_ode_lotkavoltera
    for gamma in (7//10, 9//10, 0.95),
        EEst in (0.1, 1, 10),
        dt in (0.01, 0.1, 1.)

        integ = init(prob, EKF0(), steprule=:standard, gamma=gamma)

        integ.EEst = EEst
        integ.dt = dt

        @unpack q = integ.constants
        @unpack qmin, qmax = integ.opts

        scale = (1/EEst)^(1/(q+1))
        dt_new = dt * scale * gamma
        dt_new = max(dt*qmin, min(dt*qmax, dt_new))

        dt_prop = ProbNumODE.propose_step!(integ.steprule, integ)
        @test dt_new ≈ dt_prop atol=1e-4
    end
end
