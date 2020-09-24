function multitime(callable; numruns=20, seconds=2)
    benchmark_f = () -> @elapsed callable()
    runtime = benchmark_f()
    if runtime < seconds
        runtime = mapreduce(j -> benchmark_f(), min, 2:numruns; init = runtime)
    end
    return runtime
end


function chi2(gaussian_estimate, actual_value)
    diff = gaussian_estimate.μ - actual_value
    chi2_vals = diff' * (gaussian_estimate.Σ \ diff)
    return chi2_vals
end


function make_savable(sol; appxsol=nothing)
    # sol = solve(args...; kwargs...)
    @assert DiffEqBase.has_analytic(sol.prob.f) || !isnothing(appxsol)

    y_ref(t) =
        DiffEqBase.has_analytic(sol.prob.f) ?
        sol.prob.f.analytic(sol.prob.u0, sol.prob.p, t) :
        appxsol(t)
    densetimes = collect(range(sol.t[1], stop=sol.t[end], length=1000))
    return (
        x=sol.x,
        t=sol.t,
        pu=sol.pu,
        u_ref=y_ref.(sol.t),
        t_dense=densetimes,
        pu_dense=sol.p(densetimes),
        u_ref_dense=y_ref.(densetimes),
    )
end


function compute_errors(savable_sol)
    # savable_sol should be an output of `make_savable`
    ssol = savable_sol
    diffs = ssol.pu.μ - ssol.u_ref
    dense_diffs = ssol.pu_dense.μ - ssol.u_ref_dense

    errors = Dict{Symbol, Float64}()

    errors[:final] = norm(mean(abs.(diffs[end])))
    errors[:final_est] = norm(mean(abs.(sqrt.(diag(ssol.pu.Σ[end])))))

    errors[:l2] = norm(sqrt.(mean.([float.(d) .^ 2 for d in diffs])))
    errors[:l2_est] = norm(sqrt.(mean.(diag.(ssol.pu.Σ))))
    errors[:l∞] = norm(maximum.([float.(d) .^ 2 for d in diffs]))
    errors[:l∞_est] = norm(maximum.(diag.(ssol.pu.Σ)))

    errors[:L2] = norm(sqrt.(mean.([float.(d) .^ 2 for d in dense_diffs])))
    errors[:L2_est] = norm(sqrt.(mean.(diag.(ssol.pu_dense.Σ))))
    errors[:L∞] = norm(maximum.([float.(d) .^ 2 for d in dense_diffs]))
    errors[:L∞_est] = norm(maximum.(diag.(ssol.pu_dense.Σ)))

    errors[:final_χ²] = chi2(ssol.pu[end], ssol.u_ref[end])
    errors[:χ²] = mean(remove_initial_nan(chi2.(ssol.pu, ssol.u_ref)))
    errors[:Χ²] = mean(remove_initial_nan(chi2.(ssol.pu_dense, ssol.u_ref_dense)))

    return errors
end


function remove_initial_nan(arr)
    @assert issorted(.!isnan.(arr))
    return arr[.!isnan.(arr)]
end
