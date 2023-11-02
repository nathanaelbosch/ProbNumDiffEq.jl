@doc raw"""
    IWP([wiener_process_dimension::Integer,] num_derivatives::Integer)

Integrated Wiener process.

**This is the recommended prior!** It is the most well-tested prior, both in this package
and in the probabilistic numerics literature in general
(see the [references](@ref references)).
It is also the prior that has the most efficient implementation.

The IWP can be created without specifying the dimension of the Wiener process,
in which case it will be inferred from the dimension of the ODE during the solve.
This is typically the preferred usage.

# In math
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= \Gamma \ \text{d}W(t).
\end{aligned}
```

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

function to_1d_sde(p::IWP)
    q = p.num_derivatives
    F_breve = diagm(1 => ones(q))
    L_breve = zeros(q + 1)
    L_breve[end] = 1.0
    return LTISDE(F_breve, L_breve)
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

function preconditioned_discretize_1d(iwp::IWP{elType}) where {elType}
    q = iwp.num_derivatives

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
function preconditioned_discretize(iwp::IWP)
    A_breve, Q_breve = preconditioned_discretize_1d(iwp)
    QR_breve = Q_breve.R

    d = iwp.wiener_process_dimension
    A = IsometricKroneckerProduct(d, Matrix(A_breve))
    QR = IsometricKroneckerProduct(d, Matrix(QR_breve))
    Q = PSDMatrix(QR)

    return A, Q
end

function discretize_1d(iwp::IWP{elType}, dt::Real) where {elType}
    q = iwp.num_derivatives

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
    d = p.wiener_process_dimension
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

function make_transition_matrices!(cache, prior::IWP, dt)
    @unpack A, Q, Ah, Qh, P, PI = cache
    make_preconditioners!(cache, dt)
    # A, Q = preconditioned_discretize(p) # not necessary since it's dt-independent
    # Ah = PI * A * P
    _matmul!(Ah, PI, _matmul!(Ah, A, P))
    # Qh = PI * Q * PI'
    fast_X_A_Xt!(Qh, Q, PI)
end
