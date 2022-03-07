function projection(d::Integer, q::Integer, ::Type{elType}=typeof(1.0)) where {elType}
    # Proj(deriv) = kron(diagm(0 => ones(elType, d)), [i==(deriv+1) ? 1 : 0 for i in 1:q+1]')

    # Slightly faster version of the above:
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
