function calc_H!(H, integ, cache)
    @unpack f = integ
    @unpack d, ddu, E1, E2 = cache

    if integ.alg isa EK0
        calc_H_EK0!(H, integ, cache)
    elseif integ.alg isa EK1
        calc_H_EK0!(H, integ, cache)
        # @assert integ.u == @view x_pred.μ[1:(q+1):end]
        OrdinaryDiffEq.calc_J!(ddu, integ, cache, true)
        _ddu = size(ddu, 2) != d ? view(ddu, 1:d, :) : ddu
        _matmul!(H, _ddu, cache.SolProj, -1.0, 1.0)
    elseif integ.alg isa DiagonalEK1
        calc_H_EK0!(H, integ, cache)
        OrdinaryDiffEq.calc_J!(ddu, integ, cache, true)
        ddu_diag = Diagonal(ddu)
        _matmul!(H, ddu_diag, cache.SolProj, -1.0, 1.0)
    else
        error("Unknown algorithm")
    end
    return nothing
end

function calc_H_EK0!(H, integ, cache)
    @unpack f = integ
    @unpack d, ddu, E1, E2 = cache

    if f isa DynamicalODEFunction
        @assert f.mass_matrix === I
        copy!(H, E2)
    else
        if f.mass_matrix === I
            copy!(H, E1)
        elseif f.mass_matrix isa UniformScaling
            copy!(H, E1)
            rmul!(H, f.mass_matrix.λ)
        else
            _matmul!(H, f.mass_matrix, E1)
        end
    end
    return nothing
end
