using ProbNumDiffEq
using Test
using LinearAlgebra

q = 3

@testset "destats.nf testing $alg" for init in (TaylorModeInit(), ClassicSolverInit()),
    alg in (
        EK0(order=q, smooth=false, initialization=init),
        EK1(order=q, smooth=false, initialization=init),
        EK1(order=q, smooth=false, initialization=init, autodiff=false),
        # EK1FDB(order=q, smooth=false, initialization=init, jac_quality=1),
    )

    f_counter = [0]
    function f(du, u, p, t; f_counter=f_counter)
        f_counter .+= 1
        du .= p .* u
        return nothing
    end
    u0 = [1.0]
    p = [-1.0]
    tspan = (0.0, 1.0)
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob, alg, save_everystep=false, dense=false)
    # @info alg sol.destats.nf f_counter[1]
    # @info sol.destats f_counter
    @test sol.destats.nf == f_counter[1]
end

@testset "SecondOrderODEProblem destats.nf testing $alg" for init in (TaylorModeInit(),),
    # ClassicSolverInit does not work for second order ODEs right now
    alg in (
        EK0(order=q, smooth=false, initialization=init),
        EK1(order=q, smooth=false, initialization=init),
        EK1(order=q, smooth=false, initialization=init, autodiff=false),
        # EK1FDB(order=q, smooth=false, initialization=init, jac_quality=1),
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
    sol = solve(prob, alg, save_everystep=false, dense=false)
    # @info alg sol.destats.nf f_counter[1]
    # @info sol.destats f_counter
    @test sol.destats.nf == f_counter[1]
end
