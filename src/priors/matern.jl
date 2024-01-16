@doc raw"""
    Matern([wiener_process_dimension::Integer=1,]
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
    d::Int
    q::Int
    lengthscale::R
end
Matern{elType}(dim, num_derivatives, lengthscale) where {elType} =
    Matern{elType,typeof(lengthscale)}(dim, num_derivatives, lengthscale)
Matern(dim, num_derivatives, lengthscale) =
    Matern{typeof(1.0)}(dim, num_derivatives, lengthscale)
Matern(num_derivatives, lengthscale) =
    Matern(1, num_derivatives, lengthscale)

remake(
    p::Matern{T};
    elType=T,
    dim=p.d,
    num_derivatives=p.q,
    lengthscale=p.lengthscale,
) where {T} = Matern{elType}(dim, num_derivatives, lengthscale)

initial_distribution(p::Matern{T}) where {T} = begin
    @unpack d, q = p
    D = d * (q + 1)
    sde = to_sde(p)
    μ0 = T <: LinearAlgebra.BlasFloat ? Array{T}(calloc, D) : zeros(T, D)
    Σ0 = PSDMatrix(plyapc(sde.F, sde.L)')
    return Gaussian(μ0, Σ0)
end

function to_sde(p::Matern)
    @unpack d, q = p
    l = p.lengthscale
    @assert l isa Number

    ν = q - 1 / 2
    λ = sqrt(2ν) / l

    F_breve = diagm(1 => ones(q))
    @. F_breve[end, :] = -binomial(q + 1, 0:q) * λ^((q+1):-1:1)

    L_breve = zeros(q + 1)
    L_breve[end] = 1.0

    F = IsometricKroneckerProduct(d, F_breve)
    L = IsometricKroneckerProduct(d, L_breve)
    return LTISDE(F, L)
end
