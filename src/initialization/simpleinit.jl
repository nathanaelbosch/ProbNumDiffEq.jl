function initial_update!(integ, cache, init::SimpleInit)
    @unpack u, f, p, t = integ
    @unpack x, Proj = cache
    du = integ.uprev

    f = _unwrap_f(f)

    f(du, u, p, t)
    integ.stats.nf += 1

    if f isa DynamicalODEFunction
        @assert u isa ArrayPartition
        u = u.x[2]
        @assert du isa ArrayPartition
        du = du.x[2]
    end

    update_on_data!(x, Proj(0), view(u, :), nothing; cache)
    H, rows = derivative_observation(cache, f.mass_matrix, 1)
    isempty(rows) || update_on_data!(x, H, view(du, rows), nothing; cache)
end
