using ProbNumDiffEq, Plots, LinearAlgebra


############################################################################################
# Problem and solution
############################################################################################
du0 = [0.0]
u0 = [2.0]
tspan = (0.0, 3.0)
p = [1e0]

evaltspan = (0.0, 5.0)

function vanderpol!(ddu, du, u, p, t)
    μ = p[1]
    @. ddu = μ * ((1 - u^2) * du - u)
end
prob = SecondOrderODEProblem(vanderpol!, du0, u0, tspan, p)
ref = solve(remake(prob, tspan=evaltspan), EK1(), abstol=1e-9, reltol=1e-9);
sol = solve(prob, EK1(prior=Matern(2, 1.5), diffusionmodel=FixedDiffusion()));

############################################################################################
# Sampling
############################################################################################
interp(sol, ts) = StructArray([
    ProbNumDiffEq.interpolate(
        t,
        sol.t,
        sol.x_filt,
        sol.x_smooth,
        sol.diffusions,
        sol.cache;
        smoothed=sol.alg.smooth,
    )
    for t in ts
])

function sample(sol, ts)
    states = interp(sol, ts)
    s = ProbNumDiffEq.sample_states(ts, states, sol.diffusions, sol.t, sol.cache)[:, :, 1]
    H = vcat(sol.cache.E0, sol.cache.E1, sol.cache.E2)
    return s * H'
end

ts = range(evaltspan..., length=5000)
N = 5
samples = [sample(sol, ts) for _ in 1:N]


############################################################################################
# Plotting
############################################################################################
COLORS = (
    red="#CB3C33",
    green="#389826",
    purple="#9558B2",
    blue="#4063D8",
)

vecvec2mat(vv) = hcat(vv...)'

plot(size=(900, 300), grid=false, ticks=false)

# plot reference
refH = vcat(ref.cache.E0, ref.cache.E1, ref.cache.E2);
refm = vecvec2mat(ref.x_smooth.μ) * refH';
plot!(ref.t, refm, color=:black, linestyle=:dash, label="")

# plot vline to show extrapolation
vline!(sol.t[end:end], color=:black, linewidth=1, label=false)

# plot ribbon
xs = interp(sol, ts)
H = vcat(sol.cache.E0, sol.cache.E1, sol.cache.E2)
m = vecvec2mat(xs.μ) * H'
std = sqrt.(vecvec2mat(diag.(xs.Σ))) * H'
plot!(ts, m, ribbon=3std,
    color= [COLORS[1] COLORS[2] COLORS[3]],
    label="",
    alpha=0, fillalpha=0.15)

# plot samples
plot!(ts, samples[1],
    color=COLORS,
    label=["y(t)" "ẏ(t)" "ÿ(t)"],
    linewidth=1, alpha=0.8)
for s in samples[2:N]
    for d in 3:-1:1
        plot!(ts, s[:, d], color=COLORS[d],
            label="",
            # linewidth=0.5, alpha=0.2
            alpha=0.8,
        )
    end
end

# aesthetics
plot!(
    xlims=evaltspan,
    legend=:topright,
    ylims=(-3, 5),
    dpi=1000,
)

savefig("banner.svg")
