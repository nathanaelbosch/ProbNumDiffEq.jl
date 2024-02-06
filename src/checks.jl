function check_secondorderode(integ)
    if integ.f isa DynamicalODEFunction &&
       !(integ.sol.prob.problem_type isa SecondOrderODEProblem)
        error(
            """
          The given problem is a `DynamicalODEProblem`, but not a `SecondOrderODEProblem`.
          This can not be handled by ProbNumDiffEq.jl right now. Please check if the
          problem can be formulated as a second order ODE. If not, please open a new
          github issue!
          """,
        )
    end
end
function check_densesmooth(integ)
    if integ.opts.dense && !integ.alg.smooth
        error("To use `dense=true` you need to set `smooth=true`!")
    # elseif !integ.opts.dense && integ.alg.smooth
    #     @warn "If you set dense=false for efficiency, you might also want to set smooth=false."
    end
    if !integ.opts.save_everystep && integ.alg.smooth
        error("If you do not save all values, you do not need to smooth!")
    end
end
function check_saveiter(integ)
    @assert integ.saveiter == 1
end
