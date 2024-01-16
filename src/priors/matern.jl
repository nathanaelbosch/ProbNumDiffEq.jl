@doc raw"""
    Matern([dim::Integer=1,]
           num_derivatives::Integer,
           lengthscale::Number)

Matern process.

The class of [`Matern`](@ref) processes is well-known in the Gaussian process literature,
and they also have a corresponding SDE representation similarly to the
[`IWP`](@ref) and the [`IOUP`](@ref).
See also [sarkka19appliedsde](@cite) for more details.

# In math
A Matern process is a Gauss--Markov process, which we model with a state representation
```math
\begin{aligned}
Y(t) = \left[ Y^{(0)}(t), Y^{(1)}(t), \dots Y^{(q)}(t) \right],
\end{aligned}
```
defined as the solution of the stochastic differential equation
```math
\begin{aligned}
\text{d} Y^{(i)}(t) &= Y^{(i+1)}(t) \ \text{d}t, \qquad i = 0, \dots, q-1 \\
\text{d} Y^{(q)}(t) &= - \sum_{j=0}^q \left(
  \begin{pmatrix} q+1 \\ j \end{pmatrix}
  \left( \frac{\sqrt{2q - 1}}{l} \right)^{q-j}
  Y^{(j)}(t) \right) \ \text{d}t + \Gamma \ \text{d}W(t).
\end{aligned}
```
where ``l`` is called the `lengthscale` parameter.
Then, ``Y^{(0)}(t)`` is a Matern process and
``Y^{(i)}(t)`` is the ``i``-th derivative of this process, for ``i = 1, \dots, q``.

# Examples
```julia-repl
julia> solve(prob, EK1(prior=Matern(2, 1)))
```
"""
struct Matern{elType,L} <: AbstractGaussMarkovProcess{elType}
    dim::Int
    num_derivatives::Int
    lengthscale::L
end
Matern{elType}(; dim, num_derivatives, lengthscale) where {elType} =
    Matern{elType,typeof(lengthscale)}(dim, num_derivatives, lengthscale)
Matern(; dim, num_derivatives, lengthscale) =
    Matern{typeof(1.0)}(dim, num_derivatives, lengthscale)
Matern(num_derivatives, lengthscale) =
    Matern(; dim=1, num_derivatives, lengthscale)

remake(
    p::Matern{T};
    elType=T,
    dim=dim(p),
    num_derivatives=num_derivatives(p),
    lengthscale=p.lengthscale,
) where {T} = Matern{elType}(; dim, num_derivatives, lengthscale)

initial_distribution(p::Matern{T}) where {T} = begin
    d, q = dim(p), num_derivatives(p)
    D = d * (q + 1)
    sde = to_sde(p)
    μ0 = T <: LinearAlgebra.BlasFloat ? Array{T}(calloc, D) : zeros(T, D)
    Σ0 = PSDMatrix(plyapc(sde.F, sde.L)')
    return Gaussian(μ0, Σ0)
end

function to_sde(p::Matern)
    d, q = dim(p), num_derivatives(p)
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
