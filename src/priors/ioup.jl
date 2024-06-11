@doc raw"""
    IOUP([dim::Integer=1,]
         num_derivatives::Integer,
         rate_parameter::Union{Number,AbstractVector,AbstractMatrix})

Integrated Ornstein--Uhlenbeck process.

This prior is mostly used in the context of
[Probabilistic Exponential Integrators](@ref probexpinttutorial)
to include the linear part of a semi-linear ODE in the prior,
so it is used in the [`ExpEK`](@ref) and the [`RosenbrockExpEK`](@ref).

# In math
The IOUP is a Gauss--Markov process, which we model with a state representation
```math
\begin{aligned}
Y(t) = \left[ Y^{(0)}(t), Y^{(1)}(t), \dots Y^{(q)}(t) \right],
\end{aligned}
```
defined as the solution of the stochastic differential equation
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= L Y^{(q)}(t) \ \text{d}t + \Gamma \ \text{d}W(t),
\end{aligned}
```
where ``L`` is called the `rate_parameter`.
Then, ``Y^{(0)}(t)`` is the ``q``-times integrated Ornstein--Uhlenbeck process (IOUP) and
``Y^{(i)}(t)`` is the ``i``-th derivative of the IOUP, for ``i = 1, \dots, q``.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=IOUP(2, -1)))
```
"""
struct IOUP{elType,R} <: AbstractGaussMarkovProcess{elType}
    dim::Int
    num_derivatives::Int
    rate_parameter::R
    update_rate_parameter::Bool
end
IOUP{elType}(
    ; dim, num_derivatives, rate_parameter, update_rate_parameter=false) where {elType} =
    IOUP{elType,typeof(rate_parameter)}(
        dim, num_derivatives, rate_parameter, update_rate_parameter)
IOUP(; dim=1, num_derivatives, rate_parameter, update_rate_parameter=false) =
    IOUP{typeof(1.0)}(; dim, num_derivatives, rate_parameter, update_rate_parameter)
IOUP(num_derivatives, rate_parameter; update_rate_parameter=false) =
    IOUP(; dim=1, num_derivatives, rate_parameter, update_rate_parameter)
IOUP(num_derivatives; update_rate_parameter) = begin
    @assert update_rate_parameter
    IOUP(num_derivatives, missing; update_rate_parameter)
end

initial_distribution(p::IOUP{T}) where {T} = begin
    d, q = dim(p), num_derivatives(p)
    D = d * (q + 1)
    if q > 0
        initial_variance = 1e-8 * ones(T, q + 1)
        μ0 = zeros(T, D)
        Σ0 = PSDMatrix(
            IsometricKroneckerProduct(d, diagm(sqrt.(initial_variance))),
        )
        return Gaussian(μ0, Σ0)
    else
        sde = to_sde(p)
        μ0 = zeros(T, D)
        Σ0 = PSDMatrix(
            IsometricKroneckerProduct(
                d,
                Matrix(plyapc(sde.F.B, sde.L.B)'),
            ),
        )
        return Gaussian(μ0, Σ0)
    end
end


remake(
    p::IOUP{T};
    elType=T,
    dim=dim(p),
    num_derivatives=num_derivatives(p),
    rate_parameter=p.rate_parameter,
    update_rate_parameter=p.update_rate_parameter,
) where {T} = IOUP{elType}(; dim, num_derivatives, rate_parameter, update_rate_parameter)

function to_sde(p::IOUP{T,<:Number}) where {T}
    q = num_derivatives(p)
    r = p.rate_parameter

    F_breve = diagm(1 => ones(eltype(r), q))
    F_breve[end, end] = r

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    d = dim(p)
    F = IsometricKroneckerProduct(d, F_breve)
    L = IsometricKroneckerProduct(d, L_breve)
    return LTISDE(F, L)
end
function to_sde(p::IOUP)
    d, q = dim(p), num_derivatives(p)
    r = p.rate_parameter

    R = if r isa AbstractVector
        @assert length(r) == d
        Diagonal(r)
    elseif r isa AbstractMatrix
        @assert size(r, 1) == size(r, 2) == d
        r
    end

    F_breve = diagm(1 => ones(eltype(r), q))
    # F_breve[end, end] = r
    F = kron(F_breve, Eye(d))
    F[end-d+1:end, end-d+1:end] = R

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0
    L = kron(L_breve, Eye(d))

    return LTISDE(F, L)
end

function update_sde_drift!(F::AbstractMatrix, prior::IOUP{<:Any,<:AbstractMatrix})
    d = dim(prior)
    r = prior.rate_parameter
    F[end-d+1:end, end-d+1:end] = r
end
function update_sde_drift!(F::AbstractMatrix, prior::IOUP{<:Any,<:AbstractVector})
    d = dim(prior)
    r = prior.rate_parameter
    F[end-d+1:end, end-d+1:end] = Diagonal(r)
end
function update_sde_drift!(F::AbstractMatrix, prior::IOUP{<:Any,<:Number})
    d, q = dim(prior), num_derivatives(prior)
    r = prior.rate_parameter
    F[end-d+1:end, end-d+1:end] = Diagonal(Fill(r, d))
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
