function initialize_without_derivatives(u0, f, p, t0, order, var=1e-3)
    q = order
    d = length(u0)

    m0 = zeros(d*(q+1))
    m0[1:d] = u0
    if !isinplace(f)
        m0[d+1:2d] = f(u0, p, t0)
    else
        f(m0[d+1:2d], u0, p, t0)
    end
    P0 = [zeros(d, d) zeros(d, d*q);
          zeros(d*q, d) diagm(0 => var .* ones(d*q))]
    return m0, P0
end


function __initialize_with_derivatives_forwarddiff(u0, f, p, t0, order::Int)
    error("Legacy code! ForwardDiff is not included in the dependencies anymore. Use the TaylorSeries.jl approach instead, implemented in `initialize_with_derivatives`.")
    @warn "Better to use the TaylorSeries.jl approach"
    f = isinplace(f) ? iip_to_oop(f) : f

    d = length(u0)
    q = order

    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

    m0[1:d] .= u0
    m0[d+1:2d] .= f(u0, p, t0)

    f_derivatives = Function[f]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu(u, p, t) = ForwardDiff.jacobian(u -> _curr_f_deriv(u, p, t), u)
        dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        push!(f_derivatives, df)
        m0[o*d+1:(o+1)*d] = df(u0, p, t0)
    end

    return m0, P0
end


function initialize_with_derivatives(u0, f, p, t0, order::Int)
    f = isinplace(f) ? iip_to_oop(f) : f

    d = length(u0)
    q = order

    set_variables("u", numvars=d, order=order+1)

    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

    m0[1:d] .= u0
    m0[d+1:2d] .= f(u0, p, t0)

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(t -> f(u0, p, t), t0)
    @assert !(eltype(f_t_taylor) <: TaylorN)

    fp = taylor_expand(u -> f(u, p, t0), u0)
    f_derivatives = [fp]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu = stack([derivative.(_curr_f_deriv, i) for i in 1:d])'
        # dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        # df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        df = dfdu * fp
        push!(f_derivatives, df)
        m0[o*d+1:(o+1)*d] = evaluate(df)
    end

    return m0, P0
end

function initialize_with_derivatives(u0, f::DynamicalODEFunction, p, t0, order::Int)
    f = isinplace(f) ? iip_to_oop(f) : f
    function _f(u, p, t)
        _u = ArrayPartition(u[1:1], u[2:2])
        _du = f(_u, p, t)
        du = vcat(_du...)
        return du
    end

    _u0 = vcat(u0...)
    d = length(u0)
    q = order

    set_variables("u", numvars=d, order=order+1)

    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

    m0[1:d] .= u0
    _du0 = _f(_u0, p, t0)
    m0[d+1:2d] .= _f(_u0, p, t0)

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(t -> _f(_u0, p, t), t0)
    @assert !(eltype(f_t_taylor) <: TaylorN)

    fp = taylor_expand(u -> _f(u, p, t0), _u0)
    f_derivatives = [fp]
    for o in 2:q
        _curr_f_deriv = f_derivatives[end]
        dfdu = stack([derivative.(_curr_f_deriv, i) for i in 1:d])'
        # dfdt(u, p, t) = ForwardDiff.derivative(t -> _curr_f_deriv(u, p, t), t)
        # df(u, p, t) = dfdu(u, p, t) * f(u, p, t) + dfdt(u, p, t)
        df = dfdu * fp
        push!(f_derivatives, df)
        m0[o*d+1:(o+1)*d] = evaluate(df)
    end

    return m0, P0
end

function iip_to_oop(f!)
    function f(u, p, t)
        du = copy(u)
        f!(du, u, p, t)
        return du
    end
    return f
end






# DAE STUFF
function iip_to_oop(F!::DAEFunction)
    function F(du, u, p, t)
        elType = eltype(du+u)
        out = zeros(elType, size(u))
        F!(out, du, u, p, t)
        return out
    end
    return F
end
function initialize_without_derivatives(u0, du0, f::DAEFunction, p, t0, order, var=1e-3)
    q = order
    d = length(u0)

    m0 = zeros(d*(q+1))
    m0[1:d] = u0

    f = isinplace(f) ? iip_to_oop(f) : f
    @assert iszero(f(du0, u0, p, t0))

    d = length(u0)
    q = order
    uElType = eltype(u0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1))

    m0[1:d] .= u0
    @warn "relying on `du0` for the state initialization"
    m0[d+1:2d] .= du0

    P0 = Matrix([zeros(2d, 2d) zeros(2d, d*(q-1));
          zeros(d*(q-1), 2d) Diagonal(var .* ones(d*(q-1)))])

    return m0, P0
end



function initialize_with_derivatives(u0, du0, f::DAEFunction, p, t0, order::Int)
    f = isinplace(f) ? iip_to_oop(f) : f

    d = length(u0)
    q = order

    set_variables("x", numvars=d*(q+1), order=order+1)

    uElType = eltype(u0+du0)
    m0 = zeros(uElType, d*(q+1))
    P0 = zeros(uElType, d*(q+1), d*(q+1)) .+ I(d*(q+1))

    @assert iszero(f(du0, u0, p, t0))

    m0[1:d] .= u0
    P0[1:d, 1:d] .= 0
    m0[d+1:2d] .= du0
    P0[d+1:2d, d+1:2d] .= 0

    # Make sure that the vector field f does not depend on t
    f_t_taylor = taylor_expand(t -> f(du0, u0, p, t), t0)
    @assert !(eltype(f_t_taylor) <: TaylorN)

    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    # TODO Don't redefine proj here, but use the one that we already have
    # E.g. by moving all of this into the actual `initialize!`?
    F(x) = f(Proj(1)*x, Proj(0)*x, p, t0)
    Fp = taylor_expand(m -> F(m), m0)
    driftmat = kron(diagm(1 => ones(q)), I(d))
    dm = taylor_expand(m -> driftmat*m, m0)
    Fs = [Fp]
    for o in 2:q
        F_curr = Fs[end]
        Jac_curr = stack([derivative.(F_curr, i) for i in 1:d*(q+1)])'
        m0, P0 = update_on_zero_F(m0, P0, F_curr, Jac_curr)
        dm = taylor_expand(m -> driftmat*m, m0)
        F_next = Jac_curr * dm
        push!(Fs, F_next)
    end
    F_curr = Fs[end]
    Jac_curr = stack([derivative.(F_curr, i) for i in 1:d*(q+1)])'
    m0, P0 = update_on_zero_F(m0, P0, F_curr, Jac_curr)

    return m0, P0
end
function update_on_zero_F(m, P, F, Jac; R=1e-18I)
    z = evaluate(F)
    H = evaluate(Jac')'
    S = H*P*H' + R
    _m, _P = update(Gaussian(m, P), Gaussian(z, S), H)
    return _m, _P
end
