function projection(
    d::Integer,
    q::Integer,
    ::Type{elType}=typeof(1.0),
) where {elType}
    D = d * (q + 1)
    Proj(deriv) = begin
        e_i = zeros(elType, q + 1, 1)
        if deriv <= q
            e_i[deriv+1] = 1
        end
        return kron(e_i', Eye(d))
    end
    return Proj
end
function projection(C::DenseCovariance{elType}) where {elType}
    projection(C.d, C.q, elType)
end

function projection(C::IsometricKroneckerCovariance{elType}) where {elType}
    Proj(deriv) = begin
        e_i = zeros(elType, C.q + 1, 1)
        if deriv <= C.q
            e_i[deriv+1] = 1
        end
        return IsometricKroneckerProduct(C.d, e_i')
    end
    return Proj
end

function projection(C::BlockDiagonalCovariance{elType}) where {elType}
    Proj(deriv) = begin
        e_i = zeros(elType, C.q + 1, 1)
        if deriv <= C.q
            e_i[deriv+1] = 1
        end
        return BlockDiag([copy(e_i)' for _ in 1:C.d])
    end
    return Proj
end

