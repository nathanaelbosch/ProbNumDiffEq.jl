using ProbNumDiffEq
using LinearAlgebra
using OrdinaryDiffEq: Rodas5P
using Random
using Test

# Advection, diffusion and reaction on a grid with zero boundary values. The Jacobian is
# tridiagonal and not symmetric, and the diffusion makes the problem stiff.
const N = 30
const ν, a, dx = 0.05, 0.5, 1 / N
neighbors(v, i) = (i > 1 ? v[i-1] : zero(eltype(v)), i < N ? v[i+1] : zero(eltype(v)))
function f!(du, u, p, t)
    for i in 1:N
        l, r = neighbors(u, i)
        du[i] = ν / dx^2 * (l - 2u[i] + r) - a / dx * (u[i] - l) + u[i] * (1 - u[i])
    end
    return du
end
function jvp!(Jv, v, u, p, t)
    for i in 1:N
        l, r = neighbors(v, i)
        Jv[i] = ν / dx^2 * (l - 2v[i] + r) - a / dx * (v[i] - l) + (1 - 2u[i]) * v[i]
    end
    return Jv
end
function vjp!(Jᵀw, w, u, p, t)
    for i in 1:N
        l, r = neighbors(w, i)
        Jᵀw[i] = ν / dx^2 * (l - 2w[i] + r) - a / dx * (w[i] - r) + (1 - 2u[i]) * w[i]
    end
    return Jᵀw
end
diagonal_jacobian!(out, u, p, t) = (out .= -2ν / dx^2 - a / dx .+ 1 .- 2u)

x = range(dx, 1 - dx, length=N)
u0 = @. exp(-100 * (x - 0.3)^2)
prob = ODEProblem(ODEFunction(f!; jvp=jvp!, vjp=vjp!), u0, (0.0, 1.0))
ref = solve(prob, Rodas5P(), abstol=1e-12, reltol=1e-12)
relerr(u, t) = norm(u - ref(t)) / norm(ref(t))
ek1 = solve(remake(prob, f=f!), EK1(), abstol=1e-7, reltol=1e-5)

@testset "Like the EK1 on a stiff problem: $diffusionmodel, $preconditioner" for diffusionmodel in
                                                                                 (
        DynamicDiffusion(),
        FixedDiffusion(),
    ),
    preconditioner in ("diagonal from probes", "diagonal Jacobian")

    diagonal_jacobian = preconditioner == "diagonal Jacobian" ? diagonal_jacobian! : nothing
    Random.seed!(1)
    sol = solve(
        prob,
        MatfreeEK1(; diffusionmodel, diagonal_jacobian),
        abstol=1e-7,
        reltol=1e-5,
    )
    @test sol.retcode == ReturnCode.Success
    @test relerr(sol.u[end], 1.0) < 1e-5
    @test relerr(sol(0.5).μ, 0.5) < 1e-5
    @test sol.stats.naccept < 1.5 * ek1.stats.naccept
    @test isnan(sol.pnstats.log_likelihood)
end

# With fixed steps the block-diagonal covariance limits the accuracy, more than in the EK1
@testset "Products and linear solves are counted" begin
    Random.seed!(1)
    sol = solve(prob, MatfreeEK1(), abstol=1e-7, reltol=1e-5)
    s = sol.pnstats
    @test s.njvp > 0 && s.nvjp > 0
    # The calibration in every step, the update in every accepted one
    @test s.nsolve_calibration == sol.stats.naccept + sol.stats.nreject
    @test s.nsolve_mean == s.nsolve_covariance == sol.stats.naccept
    @test sol.stats.nsolve == s.nsolve_mean + s.nsolve_covariance + s.nsolve_calibration
    @test s.niter_mean > s.nsolve_mean
    @test s.niter_covariance > 0 && s.niter_calibration > 0
    @test ek1.pnstats.njvp == ek1.pnstats.nsolve_mean == 0
end

@testset "Fixed steps" begin
    Random.seed!(1)
    sol = solve(prob, MatfreeEK1(diffusionmodel=FixedDiffusion()), adaptive=false, dt=0.01)
    @test sol.retcode == ReturnCode.Success
    @test relerr(sol.u[end], 1.0) < 1e-4
end

@testset "With iterated linearization" begin
    Random.seed!(1)
    sol = solve(prob, ODEFilter(linearization=Iterated(MatrixFreeJacobian())),
        abstol=1e-7, reltol=1e-5)
    @test sol.retcode == ReturnCode.Success
    @test relerr(sol.u[end], 1.0) < 1e-5
    @test sol.stats.nnonliniter > sol.stats.naccept
end

@testset "Out-of-place problems" begin
    f(u, p, t) = f!(similar(u), u, p, t)
    jvp(v, u, p, t) = jvp!(similar(v), v, u, p, t)
    vjp(w, u, p, t) = vjp!(similar(w), w, u, p, t)
    Random.seed!(1)
    sol = solve(ODEProblem(ODEFunction(f; jvp, vjp), u0, (0.0, 1.0)), MatfreeEK1(),
        abstol=1e-7, reltol=1e-5)
    @test relerr(sol.u[end], 1.0) < 1e-5
end

@testset "The probes are drawn from the default random number generator" begin
    sols = map((1, 1, 2)) do seed
        Random.seed!(seed)
        solve(prob, MatfreeEK1(), abstol=1e-7, reltol=1e-5)
    end
    @test sols[1].u == sols[2].u
    @test Matrix(sols[1].pu[end].Σ) == Matrix(sols[2].pu[end].Σ)
    @test Matrix(sols[1].pu[end].Σ) != Matrix(sols[3].pu[end].Σ)
end

@testset "Unsupported settings and missing inputs throw" begin
    @test_throws "`pn_observation_noise`" solve(prob, MatfreeEK1(pn_observation_noise=1.0))
    @test_throws "`diffusionmodel" solve(
        prob, MatfreeEK1(diffusionmodel=DynamicMVDiffusion()))
    @test_throws "`pn_observation_noise`" solve(
        prob,
        ODEFilter(linearization=Iterated(MatrixFreeJacobian()), pn_observation_noise=1.0))
    @test_throws ArgumentError solve(prob, MatfreeEK1(prior=IOUP(3, -1)))
    @test_throws ArgumentError solve(
        prob, MatfreeEK1(covariance_factorization=DenseCovariance))
    @test_throws "`nprobes`" solve(prob, MatfreeEK1(nprobes=2))
    @test_throws "`vjp` is missing" solve(
        ODEProblem(ODEFunction(f!; jvp=jvp!), u0, (0.0, 1.0)), MatfreeEK1())
    @test_throws "mass matrices" solve(
        ODEProblem(ODEFunction(f!; jvp=jvp!, vjp=vjp!, mass_matrix=Diagonal(ones(N))),
            u0, (0.0, 1.0)),
        MatfreeEK1())
end
