@doc raw"""
    Matern([wiener_process_dimension::Integer,]
           num_derivatives::Integer,
           lengthscale::Number)

Matern process.

As with the [`IWP`](@ref), the Matern can be created without specifying its dimension,
in which case it will be inferred from the dimension of the ODE during the solve.
This is typically the preferred usage.
The lengthscale parameter however always needs to be specified.

# In math
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= - \sum_{j=0}^q \left(
  \begin{pmatrix} q+1 \\ j \end{pmatrix}
  \left( \frac{\sqrt{2q - 1}}{l} \right)^{q-j}
  Y^{(j)}(t) \right) \ \text{d}t + \Gamma \ \text{d}W(t).
\end{aligned}
```
where ``l`` is the `lengthscale`.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=Matern(2, 1)))
```
"""
struct Matern{elType,R} <: AbstractGaussMarkovProcess{elType}
    wiener_process_dimension::Int
    num_derivatives::Int
    lengthscale::R
end
Matern(num_derivatives, lengthscale) =
    Matern(1, num_derivatives, lengthscale)
Matern(wiener_process_dimension, num_derivatives, lengthscale) =
    Matern{typeof(1.0)}(wiener_process_dimension, num_derivatives, lengthscale)
Matern{T}(wiener_process_dimension, num_derivatives, lengthscale) where {T} =
    Matern{T,typeof(wiener_process_dimension),typeof(lengthscale)}(
        wiener_process_dimension,
        num_derivatives,
        lengthscale,
    )
remake(
    p::Matern{T};
    elType=T,
    wiener_process_dimension=p.wiener_process_dimension,
    num_derivatives=p.num_derivatives,
    lengthscale=p.lengthscale,
) where {T} = Matern{elType}(wiener_process_dimension, num_derivatives, lengthscale)

initial_distribution(p::Matern{T}) where {T} = begin
    d, q = wiener_process_dimension(p), num_derivatives(p)
    D = d * (q + 1)
    sde = to_sde(p)
    μ0 = T <: LinearAlgebra.BlasFloat ? Array{T}(calloc, D) : zeros(T, D)
    Σ0 = PSDMatrix(plyapc(sde.F, sde.L)')
    return Gaussian(μ0, Σ0)
end

function to_sde(p::Matern)
    q = num_derivatives(p)
    l = p.lengthscale
    @assert l isa Number

    ν = q - 1 / 2
    λ = sqrt(2ν) / l

    F_breve = diagm(1 => ones(q))
    @. F_breve[end, :] = -binomial(q + 1, 0:q) * λ^((q+1):-1:1)

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    d = wiener_process_dimension(p)
    F = IsometricKroneckerProduct(d, F_breve)
    L = IsometricKroneckerProduct(d, L_breve)
    return LTISDE(F, L)
end
