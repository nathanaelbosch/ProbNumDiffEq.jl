function projection(
    d::Integer,
    q::Integer,
    ::Type{elType}=typeof(1.0),
) where {elType}
    D = d * (q + 1)
    Proj(deriv) = begin
        P = zeros(elType, d, D)
        @simd ivdep for i in deriv*d+1:D+1:d*D
            @inbounds P[i] = 1
        end
        return P
    end
    return Proj
end
function projection(
    ::DenseCovariance,
    d::Integer,
    q::Integer,
    ::Type{elType}=typeof(1.0),
) where {elType}
    projection(d, q, elType)
end

function projection(
    ::KroneckerCovariance,
    d::Integer,
    q::Integer,
    ::Type{elType}=typeof(1.0),
) where {elType}
    Proj(deriv) = begin
        e_i = zeros(elType, q + 1, 1)
        if deriv <= q
            e_i[deriv+1] = 1
        end
        return IsometricKroneckerProduct(d, e_i')
    end
    return Proj
end
