using ProbNumDiffEq
using OrdinaryDiffEq
using Test
using LinearAlgebra

const q = 3

@testset "stats.nf testing $alg" for alg in (
    EK0(prior=IWP(q), smooth=false),
    EK0(prior=IWP(q), smooth=false, initialization=ClassicSolverInit(alg=Tsit5())),
    EK1(prior=IWP(q), smooth=false),
    EK1(prior=IWP(q), smooth=false, autodiff=false),
    EK1(prior=IOUP(q, -1), smooth=false),
    EK1(prior=IOUP(q, update_rate_parameter=true), smooth=false),
    EK1(prior=IOUP(q, update_rate_parameter=true), smooth=true),
)
    f_counter = [0]
    function f(du, u, p, t; f_counter=f_counter)
        f_counter .+= 1
        du .= p .* u
        return nothing
    end
    u0 = [1.0]
    p = [-0.1]
    tspan = (0.0, 1.0)
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob, alg, save_everystep=alg.smooth, dense=alg.smooth)
    @test sol.stats.nf == f_counter[1]
end

@testset "SecondOrderODEProblem stats.nf testing $alg" for alg in (
    EK0(prior=IWP(q), smooth=false),
    # EK0(prior=IWP(q), smooth=false, initialization=ClassicSolverInit()),
    # ClassicSolverInit does not work for second order ODEs right now
    EK1(prior=IWP(q), smooth=false),
    EK1(prior=IWP(q), smooth=false, autodiff=false),
)
    f_counter = [0]
    function f(ddu, du, u, p, t; f_counter=f_counter)
        f_counter .+= 1
        ddu .= p .* u
        return nothing
    end
    u0 = [1.0]
    du0 = [0.0]
    p = [-1.0]
    tspan = (0.0, 1.0)
    prob = SecondOrderODEProblem(f, du0, u0, tspan, p)
    sol = solve(prob, alg, save_everystep=alg.smooth, dense=alg.smooth)
    @test sol.stats.nf == f_counter[1]
end
