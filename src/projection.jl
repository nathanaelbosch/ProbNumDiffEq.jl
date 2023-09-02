function projection(d::Integer, q::Integer, ::Type{elType}=typeof(1.0)) where {elType}
    Id = _I(d)
    Proj(deriv) = begin
        e_i = zeros(q+1, 1)
        e_i[deriv+1] = 1
        kronecker(Id, e_i')
    end
    return Proj
end
