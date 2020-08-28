function preconditioner(expected_stepsize, d, q)
    h = expected_stepsize
    diags = h .^ (-q : 0)
    I_d = diagm(0 => ones(d))
    P = Diagonal(kron(Diagonal(diags), I_d))
    P_inv = Diagonal(kron(Diagonal(1 ./ diags), I_d))
    return (P=P, P_inv=P_inv)
end
