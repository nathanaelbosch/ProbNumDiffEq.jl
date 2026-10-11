function initial_update!(integ, cache, init::SimpleInit)
    @unpack u, f, p, t = integ
    @unpack x, d, E0 = cache
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

    update!(x, linear_observation(E0, view(u, :)); cache)
    M, rows = _nonzero_rows(f.mass_matrix, d)
    H = solution_observation_matrix(cache.covariance_factorization, M, 1)
    update!(x, linear_observation(H, view(du, rows)); cache)
end
