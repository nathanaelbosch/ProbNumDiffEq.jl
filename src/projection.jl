function projection(d::Integer, q::Integer, ::Type{elType}=typeof(1.0)) where {elType}
    Proj(deriv) = begin
        e_i = zeros(q+1, 1)
        e_i[deriv+1] = 1
        kronecker(_I(d), e_i')
    end
    return Proj
end
