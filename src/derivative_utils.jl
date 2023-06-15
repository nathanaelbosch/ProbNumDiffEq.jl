function calc_H!(H, integ, cache)
    @unpack f = integ
    @unpack d, ddu, E1, E2 = cache

    if integ.alg isa EK0
        calc_H_EK0!(H, integ, cache)
    elseif integ.alg isa EK1
        calc_H_EK0!(H, integ, cache)
        # @assert integ.u == @view x_pred.μ[1:(q+1):end]
        OrdinaryDiffEq.calc_J!(ddu, integ, cache, true)
        ProbNumDiffEq._matmul!(H, view(ddu, 1:d, :), cache.SolProj, -1.0, 1.0)
    end
    return nothing
end

function calc_H_EK0!(H, integ, cache)
    @unpack f = integ
    @unpack d, ddu, E1, E2 = cache

    return nothing

    if f isa DynamicalODEFunction
        @assert f.mass_matrix === I
        H .= E2
    else
        if f.mass_matrix === I
            H .= E1
        elseif f.mass_matrix isa UniformScaling
            H .= f.mass_matrix.λ .* E1
        else
            _matmul!(H, f.mass_matrix, E1)
        end
    end
    return nothing
end

get_H(alg::EK1, cache) = cache.H
get_H(alg::EK0, cache) = cache.E1
get_H(cache) = cache.E1
