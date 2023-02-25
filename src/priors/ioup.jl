"""
    IOUP([wiener_process_dimension::Integer,]
         num_derivatives::Integer,
         rate_parameter::Union{Number,AbstractVector,AbstractMatrix})

Integrated Ornstein--Uhlenbeck process.

As with the [`IWP`](@ref), the IOUP can be created without specifying its dimension,
in which case it will be inferred from the dimension of the ODE during the solve.
This is typically the preferred usage.
The rate parameter however always needs to be specified.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=IOUP(2, -1)))
```
"""
struct IOUP{elType,dimType,R} <: AbstractODEFilterPrior{elType}
    wiener_process_dimension::dimType
    num_derivatives::Int
    rate_parameter::R
end
IOUP(num_derivatives, rate_parameter) =
    IOUP(missing, num_derivatives, rate_parameter)
IOUP(wiener_process_dimension, num_derivatives, rate_parameter) =
    IOUP{typeof(1.0)}(wiener_process_dimension, num_derivatives, rate_parameter)
IOUP{T}(wiener_process_dimension, num_derivatives, rate_parameter) where {T} =
    IOUP{T,typeof(wiener_process_dimension),typeof(rate_parameter)}(
        wiener_process_dimension,
        num_derivatives,
        rate_parameter,
    )

function to_1d_sde(p::IOUP)
    q = p.num_derivatives
    r = p.rate_parameter
    if !(r isa Number)
        m = "The rate parameter must be a scalar to convert the IOUP to a 1D SDE."
        throw(ArgumentError(m))
    end

    F_breve = diagm(1 => ones(q))
    F_breve[end, end] = r

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    return LTISDE(F_breve, L_breve)
end
function to_sde(p::IOUP)
    d = p.wiener_process_dimension
    q = p.num_derivatives
    r = p.rate_parameter

    if r isa Number
        d = p.wiener_process_dimension
        _sde = to_1d_sde(p)
        F_breve, L_breve = drift(_sde), dispersion(_sde)
        F = kron(I(d), F_breve)
        L = kron(I(d), L_breve)
        return LTISDE(F, L)
    end

    R = if r isa AbstractVector
        @assert length(r) == d
        Diagonal(r)
    elseif r isa AbstractMatrix
        @assert size(r, 1) == size(r, 2) == d
        r
    end

    F_breve = diagm(1 => ones(q))
    # F_breve[end, end] = r
    F = kron(I(d), F_breve)
    F[q+1:q+1:end, q+1:q+1:end] = R

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0
    L = kron(I(d), L_breve)

    return LTISDE(F, L)
end

function discretize(p::IOUP, dt::Real)
    r = p.rate_parameter
    A, Q = if p.rate_parameter isa Number
        A_breve, Q_breve = discretize(to_1d_sde(p), dt)
        d = p.wiener_process_dimension
        # QR_breve = cholesky!(Symmetric(Q_breve)).L'
        E = eigen(Symmetric(Q_breve))
        QR_breve = Diagonal(sqrt.(max.(E.values, 0))) * E.vectors'

        A = kron(I(d), A_breve)
        QR = kron(I(d), QR_breve)
        Q = PSDMatrix(QR)
        A, Q
    else
        @assert r isa AbstractVector || r isa AbstractMatrix
        A, Q = discretize(to_sde(p), dt)
        E = eigen(Symmetric(Q))
        QR = Diagonal(sqrt.(max.(E.values, 0))) * E.vectors'
        Q = PSDMatrix(QR)
        A, Q
    end

    return A, Q
end
