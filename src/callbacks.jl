function ManifoldUpdate(residualf)
    condition(u, t, integ) = true

    function affect!(integ)
        @unpack u = integ
        @unpack x, SolProj = integ.cache

        f(m) = residualf(SolProj * m)

        m, C = x
        z = f(m)
        J = ForwardDiff.jacobian(f, m)
        S = X_A_Xt(C, J)

        x_out = update(x, Gaussian(z, S), J)
        copy!(x, x_out)
    end
    DiscreteCallback(condition, affect!)
end
