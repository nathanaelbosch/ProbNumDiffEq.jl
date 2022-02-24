
function DiffEqBase.__init(
    prob::DiffEqBase.AbstractODEProblem{uType,tType,false},
    alg::GaussianODEFilter,
    args...;
    kwargs...,
) where {uType,tType}
    @warn "The given problem is in out-of-place form. Since the algorithms in this package are written for in-place problems, it will be automatically converted."
    if prob.f isa DynamicalODEFunction
        f1!(dv, v, u, p, t) = dv .= prob.f.f1(v, u, p, t)
        f2!(du, v, u, p, t) = du .= prob.f.f2(v, u, p, t)
        _prob = DynamicalODEProblem(
            f1!,
            f2!,
            prob.u0.x[1],
            prob.u0.x[2],
            prob.tspan,
            prob.p;
            prob.kwargs...,
        )
    else
        f!(du, u, p, t) = du .= prob.f(u, p, t)
        _prob = ODEProblem(
            ODEFunction(
                f!,
                jac=isnothing(prob.f.jac) ? nothing :
                    (ddu, u, p, t) -> ddu .= prob.f.jac(u, p, t),
                analytic=prob.f.analytic,
            ),
            prob.u0,
            prob.tspan,
            prob.p;
            prob.kwargs...,
        )
    end
    return DiffEqBase.__init(_prob, alg, args...; kwargs...)
end
