# Benchmark suite for AirspeedVelocity.jl; see performance/README.md.
#
# The same script runs on old releases, so it uses only the public API, in the spelling
# that has been stable the longest. An entry whose trial solve fails, for example because
# the solver or keyword does not exist yet in that version, is skipped with a warning.
# Set PNDE_BENCH_STRICT=1 to rethrow instead.

using BenchmarkTools: BenchmarkGroup, @benchmarkable
using LinearAlgebra: BLAS, Diagonal, dot
using ProbNumDiffEq

BLAS.set_num_threads(1)
@info "Benchmark settings" Threads.nthreads() BLAS.get_num_threads()

const STRICT = get(ENV, "PNDE_BENCH_STRICT", "0") == "1"
const SECONDS = 2

const SUITE = BenchmarkGroup()

function add!(group, name, prob, make_alg; dt, kwargs...)
    for (key, smooth) in (("filter", false), ("smoother", true))
        run = try
            alg = make_alg(smooth)
            f = () -> solve(prob, alg; adaptive=false, dt, dense=smooth, kwargs...)
            sol = f()
            string(sol.retcode) == "Success" ||
                error("trial solve returned retcode $(sol.retcode)")
            f
        catch e
            STRICT && rethrow()
            @warn "Skipping $(only(group.tags))/$name/$key" exception =
                (e, catch_backtrace())
            continue
        end
        group[name][key] = @benchmarkable $run() seconds = SECONDS
    end
    return group
end

# Problems

function lotkavolterra!(du, u, p, t)
    a, b, c, d = p
    du[1] = a * u[1] - b * u[1] * u[2]
    du[2] = -c * u[2] + d * u[1] * u[2]
    return nothing
end
const LV = ODEProblem(lotkavolterra!, [1.0, 1.0], (0.0, 10.0), [1.5, 1.0, 3.0, 1.0])

function vanderpol!(du, u, p, t)
    du[1] = u[2]
    du[2] = p[1] * ((1 - u[1]^2) * u[2] - u[1])
    return nothing
end
const VDP = ODEProblem(vanderpol!, [2.0, 0.0], (0.0, 6.3), [10.0])

const PLEIADES_MASSES = 1:7
const PLEIADES_Q0 = [3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0, 3.0, -3.0, 2.0, 0, 0, -4.0, 4.0]
const PLEIADES_V0 = [0, 0, 0, 0, 0, 1.75, -1.5, 0, 0, 0, -1.25, 1, 0, 0.0]
function pleiades_acc!(dv, q)
    x, y = view(q, 1:7), view(q, 8:14)
    fill!(dv, 0)
    for i in 1:7, j in 1:7
        i == j && continue
        dx, dy = x[j] - x[i], y[j] - y[i]
        r3 = (dx^2 + dy^2)^(3 / 2)
        dv[i] += PLEIADES_MASSES[j] * dx / r3
        dv[7+i] += PLEIADES_MASSES[j] * dy / r3
    end
    return nothing
end
function pleiades!(du, u, p, t)
    du[1:14] .= view(u, 15:28)
    pleiades_acc!(view(du, 15:28), view(u, 1:14))
    return nothing
end
pleiades2!(ddq, dq, q, p, t) = pleiades_acc!(ddq, q)
const PLEIADES = ODEProblem(pleiades!, [PLEIADES_Q0; PLEIADES_V0], (0.0, 3.0))
const PLEIADES2 = SecondOrderODEProblem(pleiades2!, PLEIADES_V0, PLEIADES_Q0, (0.0, 3.0))

const L96_N, L96_F = 50, 8.0
l96(i) = mod1(i, L96_N)
function lorenz96!(du, u, p, t)
    for i in 1:L96_N
        du[i] = (u[l96(i + 1)] - u[l96(i - 2)]) * u[l96(i - 1)] - u[i] + L96_F
    end
    return nothing
end
function lorenz96_jvp!(Jv, v, u, p, t)
    for i in 1:L96_N
        Jv[i] =
            (v[l96(i + 1)] - v[l96(i - 2)]) * u[l96(i - 1)] +
            (u[l96(i + 1)] - u[l96(i - 2)]) * v[l96(i - 1)] - v[i]
    end
    return Jv
end
function lorenz96_vjp!(Jtw, w, u, p, t)
    for j in 1:L96_N
        Jtw[j] =
            w[l96(j - 1)] * u[l96(j - 2)] - w[l96(j + 2)] * u[l96(j + 1)] +
            w[l96(j + 1)] * (u[l96(j + 2)] - u[l96(j - 1)]) - w[j]
    end
    return Jtw
end
const L96_U0 = [L96_F + (i == 1 ? 0.01 : 0.0) for i in 1:L96_N]
const L96 = ODEProblem(lorenz96!, L96_U0, (0.0, 1.0))
const L96_MATFREE = try
    ODEProblem(ODEFunction(lorenz96!; jvp=lorenz96_jvp!, vjp=lorenz96_vjp!), L96_U0, (0.0, 1.0))
catch
    L96
end

function rober!(du, u, p, t)
    k1, k2, k3 = p
    du[1] = -k1 * u[1] + k3 * u[2] * u[3]
    du[2] = k1 * u[1] - k3 * u[2] * u[3] - k2 * u[2]^2
    du[3] = u[1] + u[2] + u[3] - 1
    return nothing
end
const ROBER = ODEProblem(
    ODEFunction(rober!, mass_matrix=Diagonal([1.0, 1.0, 0.0])),
    [1.0, 0.0, 0.0], (0.0, 1e-2), (0.04, 3e7, 1e4))

function oscillator!(du, u, p, t)
    du[1] = u[2]
    du[2] = -u[1]
    return nothing
end
const OSCILLATOR = ODEProblem(oscillator!, ones(2), (0.0, 10.0))

# Noise-free data from a classic RK4 solve, so that it does not depend on the version.
function lotkavolterra_data()
    f(u) = (du=similar(u); lotkavolterra!(du, u, LV.p, 0.0); du)
    u, h, t, data_t, data_u = copy(LV.u0), 1e-3, 0.0, Float64[], Vector{Float64}[]
    for k in 1:10_000
        k1 = f(u)
        k2 = f(u + h / 2 * k1)
        k3 = f(u + h / 2 * k2)
        k4 = f(u + h * k3)
        u = u + h / 6 * (k1 + 2k2 + 2k3 + k4)
        if k % 100 == 0
            push!(data_t, k * h)
            push!(data_u, copy(u))
        end
    end
    return (t=data_t, u=data_u)
end

# Suite

SUITE["lotkavolterra"] = let g = BenchmarkGroup(["lotkavolterra"]), p = LV, dt = 0.01
    add!(g, "EK0", p, s -> EK0(smooth=s); dt)
    add!(g, "EK1", p, s -> EK1(smooth=s); dt)
    add!(g, "DiagonalEK1", p, s -> DiagonalEK1(smooth=s); dt)
    add!(g, "EK0_DynamicMVDiffusion", p,
        s -> EK0(smooth=s, diffusionmodel=DynamicMVDiffusion()); dt)
    add!(g, "EK0_FixedMVDiffusion", p,
        s -> EK0(smooth=s, diffusionmodel=FixedMVDiffusion()); dt)
    add!(g, "EK1_IOUP", p, s -> EK1(smooth=s, prior=IOUP(3, -1.0)); dt)
    add!(g, "EK1_Matern", p, s -> EK1(smooth=s, prior=Matern(3, 1.0)); dt)
end

SUITE["vanderpol"] = let g = BenchmarkGroup(["vanderpol"]), p = VDP, dt = 1e-3
    add!(g, "EK1", p, s -> EK1(smooth=s); dt)
    add!(g, "ExpEK", p, s -> ExpEK(L=-1.0, smooth=s); dt)
    add!(g, "RosenbrockExpEK", p, s -> RosenbrockExpEK(smooth=s); dt)
end

SUITE["pleiades"] = let g = BenchmarkGroup(["pleiades"]), p = PLEIADES, dt = 0.01
    add!(g, "EK0", p, s -> EK0(smooth=s); dt)
    add!(g, "EK1", p, s -> EK1(smooth=s); dt)
    add!(g, "DiagonalEK1", p, s -> DiagonalEK1(smooth=s); dt)
end

SUITE["pleiades_secondorder"] =
    let g = BenchmarkGroup(["pleiades_secondorder"]), p = PLEIADES2, dt = 0.01
        add!(g, "EK0", p, s -> EK0(smooth=s); dt)
        add!(g, "EK1", p, s -> EK1(smooth=s); dt)
    end

SUITE["lorenz96"] = let g = BenchmarkGroup(["lorenz96"]), p = L96, dt = 0.01
    add!(g, "EK0", p, s -> EK0(smooth=s); dt)
    add!(g, "EK1", p, s -> EK1(smooth=s); dt)
    add!(g, "DiagonalEK1", p, s -> DiagonalEK1(smooth=s); dt)
    add!(g, "EK0_DynamicMVDiffusion", p,
        s -> EK0(smooth=s, diffusionmodel=DynamicMVDiffusion()); dt)
    add!(g, "MatfreeEK1", L96_MATFREE, s -> MatfreeEK1(smooth=s); dt)
end

SUITE["rober_massmatrix"] =
    let g = BenchmarkGroup(["rober_massmatrix"]), p = ROBER, dt = 1e-5
        add!(g, "EK1", p, s -> EK1(smooth=s); dt)
    end

SUITE["lotkavolterra_data"] =
    let g = BenchmarkGroup(["lotkavolterra_data"]), p = LV, dt = 0.01
        cb = try
            DataUpdateCallback(lotkavolterra_data(); observation_noise_cov=1e-2)
        catch e
            STRICT && rethrow()
            @warn "Skipping lotkavolterra_data" exception = e
            nothing
        end
        if !isnothing(cb)
            add!(g, "EK0", p, s -> EK0(smooth=s); dt, callback=cb)
            add!(g, "EK1", p, s -> EK1(smooth=s); dt, callback=cb)
        end
        g
    end

SUITE["oscillator_manifold"] =
    let g = BenchmarkGroup(["oscillator_manifold"]), p = OSCILLATOR, dt = 0.01
        E(u) = [dot(u, u) - 2]
        cb = try
            ManifoldUpdate(E)
        catch e
            STRICT && rethrow()
            @warn "Skipping oscillator_manifold" exception = e
            nothing
        end
        isnothing(cb) || add!(g, "EK1", p, s -> EK1(smooth=s); dt, callback=cb)
        g
    end
