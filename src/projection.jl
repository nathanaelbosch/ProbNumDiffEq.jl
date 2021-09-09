function projection(d, q, elType=typeof(1.0))
    Proj(deriv) = kron(diagm(0 => ones(elType, d)), [i==(deriv+1) ? 1 : 0 for i in 1:q+1]')
    return Proj
end
