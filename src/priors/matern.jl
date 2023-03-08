"""
    Matern([wiener_process_dimension::Integer,]
           num_derivatives::Integer,
           lengthscale::Number)

Matern process.

As with the [`IWP`](@ref), the Matern can be created without specifying its dimension,
in which case it will be inferred from the dimension of the ODE during the solve.
This is typically the preferred usage.
The lengthscale parameter however always needs to be specified.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=Matern(2, 1)))
```
"""
struct Matern{elType,dimType,R} <: AbstractODEFilterPrior{elType}
    wiener_process_dimension::dimType
    num_derivatives::Int
    lengthscale::R
end
Matern(num_derivatives, lengthscale) =
    Matern(missing, num_derivatives, lengthscale)
Matern(wiener_process_dimension, num_derivatives, lengthscale) =
    Matern{typeof(1.0)}(wiener_process_dimension, num_derivatives, lengthscale)
Matern{T}(wiener_process_dimension, num_derivatives, lengthscale) where {T} =
    Matern{T,typeof(wiener_process_dimension),typeof(lengthscale)}(
        wiener_process_dimension,
        num_derivatives,
        lengthscale,
    )

function to_1d_sde(p::Matern)
    q = p.num_derivatives
    l = p.lengthscale

    ν = q - 1 / 2
    λ = sqrt(2ν) / l

    F_breve = diagm(1 => ones(q))
    @. F_breve[end, :] = -binomial(q + 1, 0:q) * λ^((q+1):-1:1)

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    return LTISDE(F_breve, L_breve)
end
function to_sde(p::Matern)
    d = p.wiener_process_dimension
    _sde = to_1d_sde(p)
    F_breve, L_breve = drift(_sde), dispersion(_sde)
    F = kron(I(d), F_breve)
    L = kron(I(d), L_breve)
    return LTISDE(F, L)
end
function discretize(p::Matern, dt::Real)
    l = p.lengthscale
    @assert l isa Number
    A, Q = begin
        A_breve, Q_breve = discretize(to_1d_sde(p), dt)
        d = p.wiener_process_dimension
        # QR_breve = cholesky!(Symmetric(Q_breve)).L'
        E = eigen(Symmetric(Q_breve))
        QR_breve = Diagonal(sqrt.(max.(E.values, 0))) * E.vectors'

        A = kron(I(d), A_breve)
        QR = kron(I(d), QR_breve)
        Q = PSDMatrix(QR)
        A, Q
    end

    return A, Q
end
