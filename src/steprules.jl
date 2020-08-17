abstract type StepController end
struct ConstantSteps <: StepController end
function propose_step(::ConstantSteps, integ)
    return integ.dt
end


struct StandardSteps <: StepController end
function propose_step(::StandardSteps, integ)

    @unpack dt, EEst = integ
    @unpack q = integ.constants
    @unpack gamma, qmin, qmax = integ.opts

    if iszero(EEst)
        q = inv(qmax)
    else
        localconvrate = q+1
        qtmp = DiffEqBase.fastpow(EEst, 1/localconvrate) / gamma
        @fastmath q = DiffEqBase.value(max(inv(qmax), min(inv(qmin), qtmp)))
    end
    dt /= q
    return dt
end

struct PISteps <: StepController end
function propose_step(::PISteps, integ)
    # PI-controller
    @unpack dt, EEst, qold, q11 = integ
    @unpack beta1, beta2, qmin, qmax = integ.opts

    if iszero(EEst)
        q = inv(qmax)
    else
        q11 = DiffEqBase.value(DiffEqBase.fastpow(EEst, beta1))
        q = q11 / DiffEqBase.fastpow(qold, beta2) / gamma
        integrator.q11 = q11
        @fastmath q = DiffEqBase.value(max(inv(qmax),min(inv(qmin), q)))
    end
    dt /= q
    return dt
end
