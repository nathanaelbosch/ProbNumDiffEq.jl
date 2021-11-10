using ProbNumDiffEq
using Test
using LinearAlgebra


@testset "destats.nf testing $alg" for
    q in (1, 2, 3, 5),
    init = (TaylorModeInit(), ClassicSolverInit()),
    alg in (EK0(order=q, smooth=false, initialization=init),
            EK1(order=q, smooth=false, initialization=init),
            EK1(order=q, smooth=false, initialization=init, autodiff=false),
            # EK1FDB(order=q, smooth=false, initialization=init, jac_quality=1),
            )

    f_counter = [0]
    function f(du, u, p, t; f_counter=f_counter)
        f_counter .+= 1
        mul!(du, p, u)
    end
    u0 = [1]
    p = [-1]
    tspan = (0.0, 3.0)
    prob = ODEProblem(f, u0, tspan, p)
    sol = solve(prob, alg, save_everystep=false)
    # @info alg sol.destats.nf f_counter[1]
    # @info sol.destats f_counter
    @test sol.destats.nf == f_counter[1]
end
