abstract type StepController end
struct ConstantSteps <: StepController end
function propose_step!(::ConstantSteps, integ)
    return integ.dt
end


struct StandardSteps <: StepController end
function propose_step!(::StandardSteps, integ)

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

"""Note: Setting beta1=1/(q+1), beta2=0 recovers the proportional control"""
struct PISteps <: StepController end
function propose_step!(::PISteps, integ)
    # PI-controller
    @unpack dt, EEst, qold = integ
    @unpack gamma, beta1, beta2, qmin, qmax = integ.opts

    if iszero(EEst)
        q = inv(qmax)
    elseif EEst < 1  # Accepted
        q11 = DiffEqBase.value(DiffEqBase.fastpow(EEst, beta1))
        q = q11 / DiffEqBase.fastpow(qold, beta2) / gamma

        @fastmath q = DiffEqBase.value(max(inv(qmax),min(inv(qmin), q)))
        integ.qold = max(integ.EEst, integ.opts.qoldinit)
    else # Rejected step
        q11 = DiffEqBase.value(DiffEqBase.fastpow(EEst, beta1))
        q = q11 / gamma
        @fastmath q = DiffEqBase.value(max(inv(qmax),min(inv(qmin), q)))
    end
    dt /= q
    return dt
end
