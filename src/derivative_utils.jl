function calc_H!(H, integ, cache)
    @unpack f = integ
    @unpack d, ddu, E1, E2 = cache

    if integ.alg isa EK0
        calc_H_EK0!(H, integ, cache)
    elseif integ.alg isa EK1
        calc_H_EK0!(H, integ, cache)
        # @assert integ.u == @view x_pred.μ[1:(q+1):end]
        OrdinaryDiffEq.calc_J!(ddu, integ, cache, true)
        _matmul!(H, view(ddu, 1:d, :), cache.SolProj, -1.0, 1.0)
    elseif integ.alg isa DiagonalEK1
        calc_H_EK0!(H, integ, cache)
        # @assert integ.u == @view x_pred.μ[1:(q+1):end]
        # ddu_full = Matrix(ddu)
        # @info "ddu" ddu_full
        # error()
        OrdinaryDiffEq.calc_J!(ddu, integ, cache, true)

        @unpack C_dxd = cache
        if C_dxd isa MFBD
            @simd ivdep for i in eachindex(blocks(C_dxd))
                @assert length(C_dxd.blocks[i]) == 1
                C_dxd.blocks[i][1] = ddu[i, i]
            end
        else
            C_dxd .= Diagonal(ddu)
        end
        _matmul!(H, C_dxd, cache.SolProj, -1.0, 1.0)
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
