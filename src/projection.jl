function projection(d::Integer, q::Integer, ::Type{elType}=typeof(1.0)) where {elType}
    Proj(deriv) = begin
        e_i = zeros(elType, q+1, 1)
        if deriv <= q
            e_i[deriv+1] = 1
        end
        IsoKroneckerProduct(d, e_i')
    end
    return Proj
end
