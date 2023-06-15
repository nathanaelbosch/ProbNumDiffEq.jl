function projection(d::Integer, q::Integer, ::Type{elType}=typeof(1.0)) where {elType}
    Id = _I(d)
    Proj(deriv) = begin
        e_i = zeros(q+1, 1)
        e_i[deriv+1] = 1
        kronecker(Id, e_i')
    end

    # Slightly faster version of the above:
    # D = d * (q + 1)
    # Proj(deriv) = begin
    #     P = zeros(elType, d, D)
    #     @simd ivdep for i in deriv*d+1:D+1:d*D
    #         @inbounds P[i] = 1
    #     end
    #     return P
    # end
    return Proj
end
