# Everytime I encounter something that raises some error and I fix it, I should add that
# specific problem to this list to make sure, that this specific run then works without
# bugs.
using ODEFilters
using OrdinaryDiffEq
using Test
using LinearAlgebra


########################################################################################
# Setup as in exploration.jl
########################################################################################
@testset "Henon-Heiles: Manifold projection improves the result" begin

    initial = [0.5, 0, 0, 0.1]
    tspan = (0, 100.)
    V(x,y) = 1 // 2 * (x^2 + y^2 + 2x^2 * y - 2 // 3 * y^3)  # Potential
    E(dx,dy,x,y) = V(x, y) + 1 // 2 * (dx^2 + dy^2);  # Total energy of the system
    function Hénon_Heiles(du, u, p, t)
        x  = u[3]
        y  = u[4]
        dx = u[1]
        dy = u[2]
        du[3] = dx
        du[4] = dy
        du[1] = -x - 2x * y
        du[2] = y^2 - y - x^2
    end
    prob = ODEProblem(Hénon_Heiles, initial, tspan)
    prob = ODEFilters.remake_prob_with_jac(prob)
    E(u) = E(u...)

    # Appxsol with DPRKN12
    function HH_acceleration!(dv, v, u, p, t)
        x, y  = u
        dx, dy = dv
        dv[1] = -x - 2x * y
        dv[2] = y^2 - y - x^2
    end
    initial_positions = [0.0,0.1]
    initial_velocities = [0.5,0.0]
    prob2 = SecondOrderODEProblem(HH_acceleration!, initial_velocities, initial_positions, tspan)
    appxsol = solve(prob2, DPRKN12(), abstol=1e-12, reltol=1e-12)

    sol1 = solve(prob, EK1(order=4))

    g(u) = E(u) - E(initial)
    sol2 = solve(prob, EK1(order=4, manifold=g))

    err1 = sol1[end] .- appxsol[end]
    err2 = sol2[end] .- appxsol[end]
    @test all(err1 .^2 > err2 .^ 2)

end
