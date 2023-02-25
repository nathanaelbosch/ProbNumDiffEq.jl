"""
    IWP([wiener_process_dimension::Integer,] num_derivatives::Integer)

Integrated Wiener process.

The IWP can be created without specifying the dimension of the Wiener process,
in which case it will be inferred from the dimension of the ODE during the solve.
This is typically the preferred usage.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=IWP(2)))
```
"""
struct IWP{elType,dimType} <: AbstractODEFilterPrior{elType}
    wiener_process_dimension::dimType
    num_derivatives::Int
end
# most convenient user-facing constructor:
IWP(num_derivatives) = IWP{typeof(1.0)}(missing, num_derivatives)
IWP{elType}(wiener_process_dimension, num_derivatives) where {elType} =
    IWP{elType,typeof(wiener_process_dimension)}(wiener_process_dimension, num_derivatives)
IWP(wiener_process_dimension, num_derivatives) =
    IWP{typeof(1.0)}(wiener_process_dimension, num_derivatives)

function preconditioned_discretize_1d(iwp::IWP{elType}) where {elType}
    q = iwp.num_derivatives

    A_breve = binomial.(q:-1:0, (q:-1:0)')
    Q_breve = Cauchy(collect(q:-1.0:0.0), collect((q+1):-1.0:1.0)) |> Matrix  # for Julia1.6

    QR_breve = cholesky(Q_breve).L'
    A_breve, QR_breve = elType.(A_breve), elType.(QR_breve)
    Q_breve = PSDMatrix(QR_breve)

    return A_breve, Q_breve
end

"""
    preconditioned_discretize(iwp::IWP)

Generate the discrete dynamics for a q-times integrated Wiener process (IWP).

The returned matrices `A::AbstractMatrix` and `Q::PSDMatrix` should be used in combination
with the preconditioners; see `./src/preconditioning.jl`.
"""
function preconditioned_discretize(iwp::IWP)
    A_breve, Q_breve = preconditioned_discretize_1d(iwp)
    QR_breve = Q_breve.R

    d = iwp.wiener_process_dimension
    A = kron(I(d), A_breve)
    QR = kron(I(d), QR_breve)
    Q = PSDMatrix(QR)

    return A, Q
end

function discretize_1d(iwp::IWP{elType}, dt::Real) where {elType}
    q = iwp.num_derivatives

    v = 0:q

    f = factorial.(v)
    A_breve = TriangularToeplitz(dt .^ v ./ f, :U) |> Matrix

    e = (2 * q + 1 .- v .- v')
    fr = reverse(f)
    Q_breve = @. dt^e / (e * fr * fr')

    QR_breve = cholesky(Q_breve).L'
    A_breve, QR_breve = elType.(A_breve), elType.(QR_breve)
    Q_breve = PSDMatrix(QR_breve)

    return A_breve, Q_breve
end

function discretize(p::IWP, dt::Real)
    A_breve, Q_breve = discretize_1d(p, dt)
    d = p.wiener_process_dimension
    A = kron(I(d), A_breve)
    QR = kron(I(d), Q_breve.R)
    Q = PSDMatrix(QR)
    return A, Q
end

function initialize_transition_matrices(p::IWP{T}, dt) where {T}
    A, Q = preconditioned_discretize(p)
    P, PI = initialize_preconditioner(p, dt)
    Ah = PI * A * P
    Qh = X_A_Xt(Q, PI)
    return A, Q, Ah, Qh, P, PI
end
