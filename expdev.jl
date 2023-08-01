using ProbNumDiffEq, Plots, LinearAlgebra
theme(:default; palette=["#4063D8", "#389826", "#9558B2", "#CB3C33"])

f(du, u, p, t) = (@. du = -u + sin(u))
u0 = [1.0]
tspan = (0.0, 20.0)
prob = ODEProblem(f, u0, tspan)

ref = solve(prob, EK1(), abstol=1e-10, reltol=1e-10)

STEPSIZE = 4
DM = FixedDiffusion() # recommended for fixed steps

sol_ek0 = solve(prob, EK0(prior=IOUP(3, -1), diffusionmodel=DM), adaptive=false,
    dt=STEPSIZE)
sol_ekl = solve(prob, EK0(prior=IOUP(3, -1), diffusionmodel=DM), adaptive=false,
    dt=STEPSIZE)
sol_ek1 = solve(prob, EK1(prior=IOUP(3, -1), diffusionmodel=DM), adaptive=false,
    dt=STEPSIZE)

plot(ref, color=:black, linestyle=:dash, label="Reference", ylims=(0.3, 1.05))
plot!(sol_ek0, denseplot=false, marker=:o, markersize=2, label="EK0", color=1)
plot!(sol_ekl, denseplot=false, marker=:o, markersize=2, label="EKL", color=2)
plot!(sol_ek1, denseplot=false, marker=:o, markersize=2, label="EK1", color=3)
