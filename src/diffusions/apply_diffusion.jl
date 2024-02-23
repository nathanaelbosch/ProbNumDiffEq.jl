"""
    apply_diffusion(Q::PSDMatrix, diffusion::Union{Number, Diagonal}) -> PSDMatrix

Apply the diffusion to the PSD transition noise covariance `Q`, return the result.
"""
apply_diffusion(
    Q::PSDMatrix,
    diffusion::Number,
) = PSDMatrix(Q.R * sqrt.(diffusion))
apply_diffusion(
    Q::PSDMatrix,
    diffusion::Diagonal{T,<:FillArrays.Fill},
) where {T} = apply_diffusion(Q, diffusion.diag.value)
apply_diffusion(
    Q::PSDMatrix{T,<:BlockDiag},
    diffusion::Diagonal{T,<:Vector},
) where {T} = PSDMatrix(
    BlockDiag([blocks(Q.R)[i] * sqrt.(diffusion.diag[i]) for i in eachindex(blocks(Q.R))]))
apply_diffusion(
    Q::PSDMatrix{T,<:Matrix},
    diffusion::Diagonal{T,<:Vector},
) where {T} = begin
    d = size(diffusion, 1)
    q = size(Q, 1) ÷ d - 1
    return PSDMatrix(Q.R * sqrt.(Kronecker.kronecker(Eye(q+1), diffusion)))
end

"""
    apply_diffusion!(Q::PSDMatrix, diffusion::Union{Number, Diagonal}) -> PSDMatrix

Apply the diffusion to the PSD transition noise covariance `Q` in place and return the result.
"""
apply_diffusion!(
    Q::PSDMatrix,
    diffusion::Diagonal{T,<:FillArrays.Fill},
) where {T} = begin
    rmul!(Q.R, sqrt.(diffusion.diag.value))
    return Q
end
apply_diffusion!(
    Q::PSDMatrix{T,<:BlockDiag},
    diffusion::Diagonal{T,<:Vector},
) where {T} = begin
    @simd ivdep for i in eachindex(blocks(Q.R))
        rmul!(blocks(Q.R)[i], sqrt(diffusion.diag[i]))
    end
    return Q
end
apply_diffusion!(
    Q::PSDMatrix,
    diffusion::Diagonal,
) = begin
    # @warn "This is not yet implemented efficiently; TODO"
    d = size(diffusion, 1)
    D = size(Q, 1)
    q = D ÷ d - 1
    # _matmul!(Q.R, Q.R, Kronecker.kronecker(sqrt.(diffusion), Eye(q + 1)))
    _matmul!(Q.R, Q.R, kron(Eye(q + 1), sqrt.(diffusion)))
    return Q
end

"""
    apply_diffusion!(out::PSDMatrix, Q::PSDMatrix, diffusion::Union{Number, Diagonal}) -> PSDMatrix

Apply the diffusion to the PSD transition noise covariance `Q` and store the result in `out`.
"""
apply_diffusion!(
    out::PSDMatrix,
    Q::PSDMatrix,
    diffusion::Number,
) = begin
    _matmul!(out.R, Q.R, sqrt.(diffusion))
    return out
end
apply_diffusion!(
    out::PSDMatrix,
    Q::PSDMatrix,
    diffusion::Diagonal{<:Number,<:FillArrays.Fill},
) = apply_diffusion!(out, Q, diffusion.diag.value)
apply_diffusion!(
    out::PSDMatrix{T,<:BlockDiag},
    Q::PSDMatrix{T,<:BlockDiag},
    diffusion::Diagonal{<:T,<:Vector},
) where {T} = begin
    @simd ivdep for i in eachindex(blocks(Q.R))
        _matmul!(blocks(out.R)[i], blocks(Q.R)[i], sqrt(diffusion.diag[i]))
    end
    return out
end
apply_diffusion!(
    out::PSDMatrix,
    Q::PSDMatrix,
    diffusion::Diagonal,
) = begin
    # @warn "This is not yet implemented efficiently; TODO"
    d = size(diffusion, 1)
    D = size(Q, 1)
    q = D ÷ d - 1
    # _matmul!(out.R, Q.R, Kronecker.kronecker(sqrt.(diffusion), Eye(q + 1)))
    _matmul!(out.R, Q.R, kron(Eye(q + 1), sqrt.(diffusion)))
    return out
end
