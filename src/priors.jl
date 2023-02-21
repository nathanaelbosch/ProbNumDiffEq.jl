abstract type AbstractODEFilterPrior{elType} end

"""
    initilize_transition_matrices!(p::AbstractODEFilterPrior)

Create all the (moslty empty) matrices that relate to the transition model.

The transition model (specified in `cache.prior`) is of the form
```math
X(t+h) \\mid X(t) \\sim \\mathcal{N} \\left( X(t+h); A(h) X(t), Q(h) \\right).
```
In addition, for improved numerical stability it computes preconditioning matrices ``P, P^{-1}`` as described in [1], as well as transition matrices
```math
\\begin{aligned}
A = P A(h) P^{-1}, \\\\
Q = P Q(h) P.\\\\
\\end{aligned}
```
This function creates matrices `A`, `Q`, `Ah`, `Qh`, `P`, `PI` to hold all these quantities, and returns them.
The matrices will then be filled with the correct values later, at each solver step, by calling [`make_transition_matrices`](@ref).

See also: [`make_transition_matrices`](@ref).

[1] N. Krämer, P. Hennig: **Stable Implementation of Probabilistic ODE Solvers** (2020)
"""
initialize_transition_matrices(p::AbstractODEFilterPrior, dt)

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
    P, PI = init_preconditioner(p.wiener_process_dimension, p.num_derivatives, T)
    make_preconditioner!(P, dt, p.wiener_process_dimension, p.num_derivatives)
    Ah = PI * A * P
    Qh = X_A_Xt(Q, PI)
    return A, Q, Ah, Qh, P, PI
end

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

    R = if r isa Number
        r * I(d)
    elseif r isa AbstractVector
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

"""
    LTISDE(F::AbstractMatrix, L::AbstractMatrix)

Linear time-invariant stochastic differential equation.

A LTI-SDE is a stochastic differential equation of the form
```math
dX_t = F X_t dt + L dW_t
```
where ``X_t`` is the state, ``W_t`` is a Wiener process, and ``F`` and ``L`` are matrices.
This `LTISDE` object holds the matrices ``F`` and ``L``.
It also provides some functionality to discretize the SDE via a matrix-fraction decomposition.
See: [`discretize(::LTISDE, ::Real)`](@ref).

In this package, the LTISDE is mostly used to implement the discretization of the [`IOUP`](@ref) prior.
"""
struct LTISDE{AT<:AbstractMatrix,BT<:AbstractVecOrMat}
    F::AT
    L::BT
end
drift(sde::LTISDE) = sde.F
dispersion(sde::LTISDE) = sde.L

discretize(sde::LTISDE, dt::Real) =
    matrix_fraction_decomposition(drift(sde), dispersion(sde), dt)
function matrix_fraction_decomposition(
    drift::AbstractMatrix,
    dispersion::AbstractVecOrMat,
    dt::Real,
)
    d = size(drift, 1)
    M = [drift dispersion*dispersion'; zero(drift) -drift']
    Mexp = exponential!(dt * M)
    A = Mexp[1:d, 1:d]
    Q = Mexp[1:d, d+1:end] * A'
    return A, Q
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

function initialize_transition_matrices(p::IOUP{T}, dt) where {T}
    Ah, Qh = discretize(p, dt)
    P, PI = init_preconditioner(p.wiener_process_dimension, p.num_derivatives, T)
    make_preconditioner!(P, dt, p.wiener_process_dimension, p.num_derivatives)
    A = P * Ah * PI
    Q = X_A_Xt(Qh, P)
    return A, Q, Ah, Qh, P, PI
end
