function projection(d, q, elType=typeof(1.0))
    Proj(deriv) =
        # deriv > q ? error("Projection called for non-modeled derivative") :
        kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(elType, d)))
    return Proj
end
