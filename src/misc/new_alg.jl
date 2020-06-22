"""Very simple implementation of the Forward Euler

My goal here was just to get a minimal example for an algorithm, that can then
be used in the `solve` function from DifferentialEquations.jl.
"""
using OrdinaryDiffEq
import OrdinaryDiffEq: OrdinaryDiffEqAlgorithm,OrdinaryDiffEqConstantCache,
    OrdinaryDiffEqMutableCache,
    alg_order, alg_cache, initialize!, perform_step!, @muladd, @unpack, @cache,
    constvalue


struct My_ALG <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export My_ALG

mutable struct My_ALGCache{uType,rateType,StageLimiter,StepLimiter,TabType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    k::rateType
    # tmp::uType
    # u₂::uType
    # fsalfirst::rateType
    # tab::TabType
end

struct My_ALGConstantCache <: OrdinaryDiffEqConstantCache
    dm  # Dynamics model
    mm  # Dynamics model
end

function My_ALGConstantCache()
    My_ALGConstantCache(0, 0)
end

# function alg_cache(alg::My_ALG,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{true})
#     tmp = similar(u)
#     u₂ = similar(u)
#     k = zero(rate_prototype)
#     fsalfirst = zero(rate_prototype)
#     tab = My_ALGConstantCache(real(uBottomEltypeNoUnits), real(tTypeNoUnits))
#     My_ALGCache(u,uprev,k,tmp,u₂,fsalfirst,alg.stage_limiter!,alg.step_limiter!,tab)
# end

function alg_cache(alg::My_ALG,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{false})
    My_ALGConstantCache()
end

function initialize!(integrator,cache::My_ALGConstantCache)
    # integrator.fsalfirst = integrator.f(integrator.uprev,integrator.p,integrator.t) # Pre-start fsal
    # integrator.destats.nf += 1
    # integrator.kshortsize = 1
    # integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

    # Avoid undefined entries if k is an array of arrays
    # integrator.fsallast = zero(integrator.fsalfirst)
    # integrator.k[1] = integrator.fsalfirst
end

@muladd function perform_step!(integrator,cache::My_ALGConstantCache,repeat_step=false)
    @unpack t,dt,uprev,u,f,p = integrator
    # @unpack a = cache

    k = f(uprev,p,t)
    u = uprev + dt * k

    # integrator.fsallast = f(u, p, t+dt) # For interpolation, then FSAL'd
    # integrator.destats.nf += 6
    # integrator.k[1] = integrator.fsalfirst
    integrator.u = u
end

# function initialize!(integrator,cache::My_ALGCache)
#     @unpack k,fsalfirst = cache
#     integrator.fsalfirst = fsalfirst
#     integrator.fsallast = k
#     integrator.kshortsize = 1
#     resize!(integrator.k, integrator.kshortsize)
#     integrator.k[1] = integrator.fsalfirst
#     integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # FSAL for interpolation
#     integrator.destats.nf += 1
# end

# @muladd function perform_step!(integrator,cache::My_ALGCache,repeat_step=false)
#     @unpack t,dt,uprev,u,f,p = integrator
#     @unpack k,tmp,u₂,fsalfirst,stage_limiter!,step_limiter! = cache
#     @unpack α40,α41,α43,α62,α65,β10,β21,β32,β43,β54,β65,c1,c2,c3,c4,c5 = cache.tab

#     f( k,  uprev, p, t+c1*dt)
#     @. u = uprev + dt * k
#     stage_limiter!(u, f, t+dt)
#     step_limiter!(u, f, t+dt)
#     integrator.destats.nf += 6
#     f( k,  u, p, t+dt)
# end


#oop test
f = ODEFunction((u,p,t)->1.01u,
                analytic = (u0,p,t) -> u0*exp(1.01t))
prob = ODEProblem(f,1.01,(0.0,1.0))
sol = solve(prob,My_ALG(),dt=0.1)

using Plots
pyplot()
p = plot(sol, denseplot=false, marker=:x)
# p = plot(sol,denseplot=false,plot_analytic=true)
savefig(p, "test.pdf")

# using DiffEqDevTools
# dts = (1/2) .^ (8:-1:1)
# sim = test_convergence(dts,prob,My_ALG())
# sim.𝒪est[:final]
# plot(sim)

# # Exanple of a good one!
# sim = test_convergence(dts,prob,BS3())
# sim.𝒪est[:final]
# plot(sim)

# #iip test
# f = ODEFunction((du,u,p,t)->(du .= 1.01.*u),
#                 analytic = (u0,p,t) -> u0*exp(1.01t))
# prob = ODEProblem(f,[1.01],(0.0,1.0))
# sol = solve(prob,My_ALG(),dt=0.1)

# plot(sol)
# plot(sol,denseplot=false,plot_analytic=true)

# dts = (1/2) .^ (8:-1:1)
# sim = test_convergence(dts,prob,My_ALG())
# sim.𝒪est[:final]
# plot(sim)

# # Exanple of a good one!
# sim = test_convergence(dts,prob,BS3())
# sim.𝒪est[:final]
# plot(sim)
