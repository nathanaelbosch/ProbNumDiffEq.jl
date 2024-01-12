@doc raw"""
    IOUP([wiener_process_dimension::Integer,]
         num_derivatives::Integer,
         rate_parameter::Union{Number,AbstractVector,AbstractMatrix})

Integrated Ornstein--Uhlenbeck process.

As with the [`IWP`](@ref), the IOUP can be created without specifying its dimension,
in which case it will be inferred from the dimension of the ODE during the solve.
This is typically the preferred usage.
The rate parameter however always needs to be specified.

# In math
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= L Y^{(q)}(t) \ \text{d}t + \Gamma \ \text{d}W(t),
\end{aligned}
```
where ``L`` is the `rate_parameter`.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=IOUP(2, -1)))
```
"""
struct IOUP{elType,dimType,R} <: AbstractGaussMarkovPrior{elType}
    wiener_process_dimension::dimType
    num_derivatives::Int
    rate_parameter::R
    update_rate_parameter::Bool
end
IOUP(num_derivatives; update_rate_parameter) = begin
    @assert update_rate_parameter
    IOUP(missing, num_derivatives, missing, update_rate_parameter)
end
IOUP(num_derivatives, rate_parameter; update_rate_parameter=false) =
    IOUP(missing, num_derivatives, rate_parameter, update_rate_parameter)
IOUP(
    wiener_process_dimension,
    num_derivatives,
    rate_parameter,
    update_rate_parameter=false,
) =
    IOUP{typeof(1.0)}(
        wiener_process_dimension,
        num_derivatives,
        rate_parameter,
        update_rate_parameter,
    )
IOUP{T}(
    wiener_process_dimension,
    num_derivatives,
    rate_parameter,
    update_rate_parameter=false,
) where {T} =
    IOUP{T,typeof(wiener_process_dimension),typeof(rate_parameter)}(
        wiener_process_dimension,
        num_derivatives,
        rate_parameter,
        update_rate_parameter,
    )

function to_sde(p::IOUP{T,D,<:Number}) where {T,D}
    q = p.num_derivatives
    r = p.rate_parameter

    F_breve = diagm(1 => ones(q))
    F_breve[end, end] = r

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    d = p.wiener_process_dimension
    F = IsometricKroneckerProduct(d, F_breve)
    L = IsometricKroneckerProduct(d, L_breve)
    return LTISDE(F, L)
end
function to_sde(p::IOUP)
    d = p.wiener_process_dimension
    q = p.num_derivatives
    r = p.rate_parameter

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
    F, L = to_sde(p)
    if F isa IsometricKroneckerProduct
        method = FiniteHorizonGramians.ExpAndGram{eltype(F.B),13}()
        A_breve, QR_breve = FiniteHorizonGramians.exp_and_gram_chol(F.B, L.B, dt, method)
        A = IsometricKroneckerProduct(F.ldim, A_breve)
        Q = PSDMatrix(IsometricKroneckerProduct(F.ldim, QR_breve))
        return A, Q
    else
        method = FiniteHorizonGramians.ExpAndGram{eltype(F),13}()
        A, QR = FiniteHorizonGramians.exp_and_gram_chol(F, L, dt, method)
        Q = PSDMatrix(QR)
        return A, Q
    end
end

function update_sde_drift!(F::AbstractMatrix, prior::IOUP{<:Any,<:Any,<:AbstractMatrix})
    q = prior.num_derivatives
    r = prior.rate_parameter
    F[q+1:q+1:end, q+1:q+1:end] = r
end
function update_sde_drift!(F::AbstractMatrix, prior::IOUP{<:Any,<:Any,<:AbstractVector})
    q = prior.num_derivatives
    r = prior.rate_parameter
    F[q+1:q+1:end, q+1:q+1:end] = Diagonal(r)
end
function update_sde_drift!(F::AbstractMatrix, prior::IOUP{<:Any,<:Any,<:Number})
    d = prior.wiener_process_dimension
    q = prior.num_derivatives
    r = prior.rate_parameter
    F[q+1:q+1:end, q+1:q+1:end] = Diagonal(Fill(r, d))
end

function make_transition_matrices!(cache, prior::IOUP, dt)
    @unpack F, L, A, Q, Ah, Qh, P, PI = cache

    update_sde_drift!(F, prior)

    make_preconditioners!(cache, dt)

    FiniteHorizonGramians.exp_and_gram_chol!(
        Ah, Qh.R, F, L, dt, cache.FHG_method, cache.FHG_cache)

    _matmul!(A, P, _matmul!(A, Ah, PI))
    fast_X_A_Xt!(Q, Qh, P)
end
