function preconditioner(expected_stepsize, d, q)
    h = expected_stepsize
    I_d = diagm(0 => ones(d))
    P = Diagonal(kron(Diagonal(h .^ (0:q)), I_d))
    P_inv = Diagonal(kron(Diagonal(1 ./ (h .^ (0:q))), I_d))
    return (P=P, P_inv=P_inv)
    # return (P=I, P_inv=I)
end
function apply_preconditioner!(p, x::Gaussian)
    x.μ .= p.P * x.μ
    x.Σ .= p.P * x.Σ * p.P'
end
function undo_preconditioner!(p, x::Gaussian)
    x.μ .= p.P_inv * x.μ
    x.Σ .= p.P_inv * x.Σ * p.P_inv'
end
function undo_preconditioner!(sol, proposals, integ)
    for s in sol
        undo_preconditioner!(integ.preconditioner, s.x)
    end
    for p in proposals
        undo_preconditioner!(integ.preconditioner, p.prediction)
        undo_preconditioner!(integ.preconditioner, p.filter_estimate)
    end
end
