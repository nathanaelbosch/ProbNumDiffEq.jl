using OrdinaryDiffEq
import OrdinaryDiffEq: OrdinaryDiffEqAlgorithm,OrdinaryDiffEqConstantCache,
      alg_order, alg_cache, initialize!, perform_step!, @muladd, @unpack, @cache,
      constvalue

struct RK_ALG <: OrdinaryDiffEq.OrdinaryDiffEqAlgorithm end
export RK_ALG
alg_order(alg::RK_ALG) = 3

@cache struct RK_ALGCache{uType,rateType,StageLimiter,StepLimiter,TabType} <: OrdinaryDiffEqMutableCache
  u::uType
  uprev::uType
  k::rateType
  tmp::uType
  u₂::uType
  fsalfirst::rateType
  stage_limiter!::StageLimiter
  step_limiter!::StepLimiter
  tab::TabType
end

struct RK_ALGConstantCache{T,T2} <: OrdinaryDiffEqConstantCache
  α40::T
  α41::T
  α43::T
  α62::T
  α65::T
  β10::T
  β21::T
  β32::T
  β43::T
  β54::T
  β65::T
  c1::T2
  c2::T2
  c3::T2
  c4::T2
  c5::T2
end

function RK_ALGConstantCache(T, T2)
  α40 = T(0.476769811285196)
  α41 = T(0.098511733286064)
  α43 = T(0.424718455428740)
  α62 = T(0.155221702560091)
  α65 = T(0.844778297439909)
  β10 = T(0.284220721334261)
  β21 = T(0.284220721334261)
  β32 = T(0.284220721334261)
  β43 = T(0.120713785765930)
  β54 = T(0.284220721334261)
  β65 = T(0.240103497065900)
  c1 = T2(0.284220721334261)
  c2 = T2(0.568441442668522)
  c3 = T2(0.852662164002783)
  c4 = T2(0.510854218958172)
  c5 = T2(0.795074940292433)

  RK_ALGConstantCache(α40, α41, α43, α62, α65, β10, β21, β32, β43, β54, β65, c1, c2, c3, c4, c5)
end

function alg_cache(alg::RK_ALG,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{true})
  tmp = similar(u)
  u₂ = similar(u)
  k = zero(rate_prototype)
  fsalfirst = zero(rate_prototype)
  tab = RK_ALGConstantCache(real(uBottomEltypeNoUnits), real(tTypeNoUnits))
  RK_ALGCache(u,uprev,k,tmp,u₂,fsalfirst,alg.stage_limiter!,alg.step_limiter!,tab)
end

function alg_cache(alg::RK_ALG,u,rate_prototype,uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits,uprev,uprev2,f,t,dt,reltol,p,calck,::Val{false})
  RK_ALGConstantCache(real(uBottomEltypeNoUnits), real(tTypeNoUnits))
end

function initialize!(integrator,cache::RK_ALGConstantCache)
  integrator.fsalfirst = integrator.f(integrator.uprev,integrator.p,integrator.t) # Pre-start fsal
  integrator.destats.nf += 1
  integrator.kshortsize = 1
  integrator.k = typeof(integrator.k)(undef, integrator.kshortsize)

  # Avoid undefined entries if k is an array of arrays
  integrator.fsallast = zero(integrator.fsalfirst)
  integrator.k[1] = integrator.fsalfirst
end

@muladd function perform_step!(integrator,cache::RK_ALGConstantCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack α40,α41,α43,α62,α65,β10,β21,β32,β43,β54,β65,c1,c2,c3,c4,c5 = cache

  # u1 -> stored as u
  u = uprev + β10 * dt * integrator.fsalfirst
  k = f(u, p, t+c1*dt)
  # u2
  u₂ = u + β21 * dt * k
  k = f(u₂,p,t+c2*dt)
  # u3
  tmp = u₂ + β32 * dt * k
  k = f(tmp, p, t+c3*dt)
  # u4
  tmp = α40 * uprev + α41 * u + α43 * tmp + β43 * dt * k
  k = f(tmp, p, t+c4*dt)
  # u5
  tmp = tmp + β54 * dt * k
  k = f(tmp, p, t+c5*dt)
  # u
  u = α62 * u₂ + α65 * tmp + β65 * dt * k

  integrator.fsallast = f(u, p, t+dt) # For interpolation, then FSAL'd
  integrator.destats.nf += 6
  integrator.k[1] = integrator.fsalfirst
  integrator.u = u
end

function initialize!(integrator,cache::RK_ALGCache)
  @unpack k,fsalfirst = cache
  integrator.fsalfirst = fsalfirst
  integrator.fsallast = k
  integrator.kshortsize = 1
  resize!(integrator.k, integrator.kshortsize)
  integrator.k[1] = integrator.fsalfirst
  integrator.f(integrator.fsalfirst,integrator.uprev,integrator.p,integrator.t) # FSAL for interpolation
  integrator.destats.nf += 1
end

@muladd function perform_step!(integrator,cache::RK_ALGCache,repeat_step=false)
  @unpack t,dt,uprev,u,f,p = integrator
  @unpack k,tmp,u₂,fsalfirst,stage_limiter!,step_limiter! = cache
  @unpack α40,α41,α43,α62,α65,β10,β21,β32,β43,β54,β65,c1,c2,c3,c4,c5 = cache.tab

  # u1 -> stored as u
  @.. u = uprev + β10 * dt * integrator.fsalfirst
  stage_limiter!(u, f, t+c1*dt)
  f( k,  u, p, t+c1*dt)
  # u2
  @.. u₂ = u + β21 * dt * k
  stage_limiter!(u₂, f, t+c2*dt)
  f(k,u₂,p,t+c2*dt)
  # u3
  @.. tmp = u₂ + β32 * dt * k
  stage_limiter!(tmp, f, t+c3*dt)
  f( k,  tmp, p, t+c3*dt)
  # u4
  @.. tmp = α40 * uprev + α41 * u + α43 * tmp + β43 * dt * k
  stage_limiter!(tmp, f, t+c4*dt)
  f( k,  tmp, p, t+c4*dt)
  # u5
  @.. tmp = tmp + β54 * dt * k
  stage_limiter!(tmp, f, t+c5*dt)
  f( k,  tmp, p, t+c5*dt)
  # u
  @.. u = α62 * u₂ + α65 * tmp + β65 * dt * k
  stage_limiter!(u, f, t+dt)
  step_limiter!(u, f, t+dt)
  integrator.destats.nf += 6
  f( k,  u, p, t+dt)
end

#oop test
f = ODEFunction((u,p,t)->1.01u,
            analytic = (u0,p,t) -> u0*exp(1.01t))
prob = ODEProblem(f,1.01,(0.0,1.0))
sol = solve(prob,RK_ALG(),dt=0.1)

using Plots
plot(sol)
plot(sol,denseplot=false,plot_analytic=true)

using DiffEqDevTools
dts = (1/2) .^ (8:-1:1)
sim = test_convergence(dts,prob,RK_ALG())
sim.𝒪est[:final]
plot(sim)

# Exanple of a good one!
sim = test_convergence(dts,prob,BS3())
sim.𝒪est[:final]
plot(sim)

#iip test
f = ODEFunction((du,u,p,t)->(du .= 1.01.*u),
            analytic = (u0,p,t) -> u0*exp(1.01t))
prob = ODEProblem(f,[1.01],(0.0,1.0))
sol = solve(prob,RK_ALG(),dt=0.1)

plot(sol)
plot(sol,denseplot=false,plot_analytic=true)

dts = (1/2) .^ (8:-1:1)
sim = test_convergence(dts,prob,RK_ALG())
sim.𝒪est[:final]
plot(sim)

# Exanple of a good one!
sim = test_convergence(dts,prob,BS3())
sim.𝒪est[:final]
plot(sim)
