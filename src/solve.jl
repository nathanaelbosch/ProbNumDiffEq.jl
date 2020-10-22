# Overwrite the OrdinaryDiffEq solve! function to call my own stuff
function DiffEqBase.solve!(integ::OrdinaryDiffEq.ODEIntegrator{<:AbstractEKF})
    tmax = integ.sol.prob.tspan[2]
    while integ.t < tmax
        loopheader!(integ)
        if check_error!(integ) != :Success
            return integ.sol
        end
        perform_step!(integ, integ.cache)
        loopfooter!(integ)
    end
    postamble!(integ)
    if integ.sol.retcode == :Default
        integ.sol = solution_new_retcode(integ.sol, :Success)
    end
    return integ.sol
end
