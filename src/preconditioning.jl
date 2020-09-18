function preconditioner(d, q)
    I_d = diagm(0 => ones(d))
    P(h) = Diagonal(kron(Diagonal(h .^ (-q : 0)), I_d))
    P_inv(h) = Diagonal(kron(Diagonal(h .^ (q : -1 : 0)), I_d))
    return (P=P, P_inv=P_inv)
end
