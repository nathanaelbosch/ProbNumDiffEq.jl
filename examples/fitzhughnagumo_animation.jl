using ProbNumDiffEq
using ProbNumDiffEq: stack
using DifferentialEquations
using Statistics
using Plots

C1, C2 = "#107D79", "#FF9933"

# ODE definition as in DifferentialEquations.jl
function fitz(u, p, t)
    a, b, c = p
    return [
        c * (u[1] - u[1]^3 / 3 + u[2])
        -(1 / c) * (u[1] - a - b * u[2])
    ]
end
u0 = [-1.0; 1.0]
tspan = (0.0, 20.0)
p = (0.2, 0.2, 3.0)
prob = ODEProblem(fitz, u0, tspan, p)

appxsol = solve(prob, abstol=1e-12, reltol=1e-12)

############################################################################################
# Solve with integrator interface and make animation
integ = init(prob, EK0(order=1), adaptive=false, dt=7e-2)

anim = @animate for i in 1:(prob.tspan[2]/integ.dt)
    step!(integ)

    plot(
        integ.sol;
        color=[C1 C2],
        ribbon_width=3,
        xlims=prob.tspan,
        ylims=(-2.5, 2.5),
        ylabel="u(t)",
        xlabel="t",
        label=["u₁(t)" "u₂(t)"],
    )

    times = integ.t:0.1:prob.tspan[2]
    post = integ.sol(times).u
    plot!(
        times,
        stack(mean(post)),
        ribbon=3stack(std(post)),
        linestyle=:dot,
        color=[C1 C2],
        fillalpha=0.2,
        label="",
    )
    scatter!(integ.sol.t, stack(integ.sol.u), color=[C1 C2], label="")
    plot!(appxsol, linestyle=:dash, color=:black, label="")
    vline!([integ.t], label="", color=:black)
end

gif(anim, "./fitzhughnagumo_solve.gif")
