@doc raw"""
    IWP([dim::Integer=1,] num_derivatives::Integer)

``q``-times integrated Wiener process.

**This is the recommended prior!**
The IWP is the most common prior choice in the probabilistic ODE solver literature
(see the [references](@ref references)) and is the default choice for the solvers in
ProbNumDiffEq.jl.
It is also the prior that has the most efficient implementation.

The IWP can be created without specifying the dimension of the Wiener process,
in which case it will be one-dimensional. The ODE solver then assumes that each dimension
of the ODE solution should be modeled with the same prior.
This is typically the preferred usage.

# In math
The IWP is a Gauss--Markov process, which we model with a state representation
```math
\begin{aligned}
Y(t) = \left[ Y^{(0)}(t), Y^{(1)}(t), \dots Y^{(q)}(t) \right],
\end{aligned}
```
defined as the solution of the stochastic differential equation
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= \Gamma \ \text{d}W(t).
\end{aligned}
```
Then, ``Y^{(0)}(t)`` is the ``q``-times integrated Wiener process (IWP) and
``Y^{(i)}(t)`` is the ``i``-th derivative of the IWP, for ``i = 1, \dots, q``.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=IWP(2)))
```
"""
struct IWP{elType} <: AbstractGaussMarkovProcess{elType}
    dim::Int
    num_derivatives::Int
end
IWP(; dim, num_derivatives) = IWP{typeof(1.0)}(dim, num_derivatives)
IWP(num_derivatives) = IWP(; dim=1, num_derivatives)

remake(
    p::IWP{T};
    elType=T,
    dim=dim(p),
    num_derivatives=num_derivatives(p),
) where {T} = IWP{elType}(dim, num_derivatives)

function to_sde(p::IWP)
    d, q = dim(p), num_derivatives(p)

    F_breve = diagm(1 => ones(q))
    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    F = IsometricKroneckerProduct(d, F_breve)
    L = IsometricKroneckerProduct(d, L_breve)
    return LTISDE(F, L)
end

"""
  make_preconditioned_transition_cov_cholf_1d(q::Integer, elType::Type)

Compute the left square root of the preconditioned IWP transition covariance.

This is based on a similar implementation in IntegratedWienerProcesses.jl
https://github.com/filtron/IntegratedWienerProcesses.jl/tree/main
adjusted such that we obtain the left square root, with different ordering of the derivatives.
"""
@fastmath function make_preconditioned_iwp_transition_cov_lsqrt_1d(
    q::Integer,
    ::Type{elType},
) where {elType}
    if q >= 10 && !(q isa BigInt)
        return make_preconditioned_iwp_transition_cov_lsqrt_1d(big(q), elType)
    end

    L = zeros(elType, q + 1, q + 1)

    @simd ivdep for m in 0:q
        @simd ivdep for n in 0:m
            # if m >= n
            @inbounds L[m+1, n+1] =
                sqrt(2q - 2m + 1) * factorial(q - n)^2 / factorial(m - n) /
                factorial(2q - n - m + 1)
            # end
        end
    end
    return L
end

function preconditioned_discretize_1d(p::IWP{elType}) where {elType}
    q = num_derivatives(p)

    A_breve = binomial.(q:-1:0, (q:-1:0)')
    # Q_breve = Cauchy(collect(q:-1.0:0.0), collect((q+1):-1.0:1.0)) |> Matrix  # for Julia1.6

    # QR_breve = cholesky(Q_breve).L' |> collect
    QR_breve = make_preconditioned_iwp_transition_cov_lsqrt_1d(q, elType)
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
function preconditioned_discretize(p::IWP)
    A_breve, Q_breve = preconditioned_discretize_1d(p)
    QR_breve = Q_breve.R

    d = dim(p)
    A = IsometricKroneckerProduct(d, Matrix(A_breve))
    QR = IsometricKroneckerProduct(d, Matrix(QR_breve))
    Q = PSDMatrix(QR)

    return A, Q
end

function discretize_1d(p::IWP{elType}, dt::Real) where {elType}
    q = num_derivatives(p)

    v = 0:q

    f = factorial.(v)
    A_breve = TriangularToeplitz(dt .^ v ./ f, :U)

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
    d = dim(p)
    A = IsometricKroneckerProduct(d, A_breve)
    QR = IsometricKroneckerProduct(d, Q_breve.R)
    Q = PSDMatrix(QR)
    return A, Q
end

function initialize_transition_matrices(FAC::IsometricKroneckerCovariance, p::IWP, dt)
    A, Q = preconditioned_discretize(p)
    P, PI = initialize_preconditioner(FAC, p, dt)
    Ah = PI * A * P
    Qh = PSDMatrix(Q.R * PI)
    return A, Q, Ah, Qh, P, PI
end
function initialize_transition_matrices(FAC::DenseCovariance, p::IWP, dt)
    A, Q = preconditioned_discretize(p)
    P, PI = initialize_preconditioner(FAC, p, dt)
    A, Q = Matrix(A), PSDMatrix(Matrix(Q.R))
    Ah, Qh = copy(A), copy(Q)
    return A, Q, Ah, Qh, P, PI
end
function initialize_transition_matrices(FAC::BlockDiagonalCovariance, p::IWP, dt)
    A, Q = preconditioned_discretize(p)
    A = to_factorized_matrix(FAC, A)
    Q = to_factorized_matrix(FAC, Q)
    P, PI = initialize_preconditioner(FAC, p, dt)
    Ah = PI * A * P
    Qh = PSDMatrix(Q.R * PI)
    return A, Q, Ah, Qh, P, PI
end

function make_transition_matrices!(cache, prior::IWP, dt)
    @unpack A, Q, Ah, Qh, P, PI = cache
    make_preconditioners!(cache, dt)
    # A, Q = preconditioned_discretize(p) # not necessary since it's dt-independent
    # Ah = PI * A * P
    _matmul!(Ah, PI, _matmul!(Ah, A, P))
    # Qh = PI * Q * PI'
    fast_X_A_Xt!(Qh, Q, PI)
end
