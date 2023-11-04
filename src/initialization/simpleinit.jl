function initial_update!(integ, cache, init::SimpleInit)
    @unpack u, f, p, t = integ
    @unpack du, x, Proj = cache

    if f isa ODEFunction &&
       f.f isa SciMLBase.FunctionWrappersWrappers.FunctionWrappersWrapper
        f = ODEFunction(SciMLBase.unwrapped_f(f), mass_matrix=f.mass_matrix)
    end

    # This is hacky and should definitely be removed. But it also works so 🤷
    MM = if f.mass_matrix isa UniformScaling
        f.mass_matrix
    else
        _MM = copy(f.mass_matrix)
        if any(iszero.(diag(_MM)))
            _MM = typeof(promote(_MM[1], 1e-20)[1]).(_MM)
            _MM .+= 1e-20I(d)
        end
        _MM
    end

    f(du, u, p, t)
    integ.stats.nf += 1

    if f isa DynamicalODEFunction
        @assert u isa ArrayPartition
        u = u[2, :]
        @assert du isa ArrayPartition
        du = du[2, :]
    end

    init_condition_on!(x, Proj(0), view(u, :), cache)
    init_condition_on!(x, MM * Proj(1), view(du, :), cache)
end
