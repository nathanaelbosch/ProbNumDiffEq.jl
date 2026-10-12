"""
    AbstractLinearization

How an [`ODEFilter`](@ref) linearizes the observation of the ODE in each step.

A linearization `L <: AbstractLinearization` has methods for
- `linearize_observation!(::L, integ, x::Gaussian)`, to linearize at the distribution `x`;
- `make_linearization_cache(::L, C, u, f)`, the buffers it writes into;
- `needs_jacobian(::L)`.
"""
abstract type AbstractLinearization end

"""
    AbstractTaylorLinearization <: AbstractLinearization

A linearization by a Taylor expansion of the ODE vector field at a point. At a `Gaussian`,
it linearizes at the mean. Its subtypes implement
`linearize_observation!(::L, integ, m::AbstractVector)` at a point `m`, instead of the
method for a `Gaussian`.
"""
abstract type AbstractTaylorLinearization <: AbstractLinearization end

"""
    ZeroJacobian()

Linearize the ODE vector field with a zero Jacobian, as the [`EK0`](@ref) does.
"""
struct ZeroJacobian <: AbstractTaylorLinearization end

"""
    DiagonalJacobian()

Linearize the ODE vector field with the diagonal of its Jacobian, as the
[`DiagonalEK1`](@ref) does.
"""
struct DiagonalJacobian <: AbstractTaylorLinearization end

"""
    FullJacobian()

Linearize the ODE vector field with its Jacobian, as the [`EK1`](@ref) does.
"""
struct FullJacobian <: AbstractTaylorLinearization end

"""
    needs_jacobian(component)

Whether a component of an `ODEFilter` needs the Jacobian of the ODE vector field.
"""
needs_jacobian(::ZeroJacobian) = false
needs_jacobian(::Union{DiagonalJacobian,FullJacobian}) = true

# The name that an `ODEFilter` with this linearization prints as
_solver_name(::ZeroJacobian) = "EK0"
_solver_name(::FullJacobian) = "EK1"
_solver_name(::DiagonalJacobian) = "DiagonalEK1"
_solver_name(::AbstractLinearization) = nothing

"""
    make_linearization_cache(linearization, C::CovarianceStructure, u, f)

The buffers that [`linearize_observation!`](@ref) writes into, for the covariance structure
`C`, the ODE state `u` and the ODE function `f`. Built once per solve, and stored in
`integ.cache.linearization_cache`.
"""
function make_linearization_cache(::ZeroJacobian, C::CovarianceStructure{T}, u, f) where {T}
    d, D = C.d, C.d * (C.q + 1)
    return (; r=similar(Array{T}, d), H=factorized_similar(C, d, D))
end
function make_linearization_cache(
    ::Union{FullJacobian,DiagonalJacobian}, C::CovarianceStructure{T}, u, f,
) where {T}
    J = isnothing(f.jac_prototype) ? zeros(T, length(u), length(u)) : f.jac_prototype
    return (; make_linearization_cache(ZeroJacobian(), C, u, f)..., J)
end

"""
    linearize_observation!(linearization, integ, x::Gaussian)
    linearize_observation!(::AbstractTaylorLinearization, integ, m::AbstractVector)

Linearize the observation of the ODE at the end of the current step at the distribution
`x`, and return it as a [`LinearizedObservation`](@ref). The observation is ``h(x) = 0``,
with ``h(x) = M E_1 x - f(E_0 x, t)``, or ``h(x) = E_2 x - f(E_1 x, E_0 x, t)`` for
second-order ODEs. A Taylor linearization linearizes at the mean of `x`, or at any point
`m`. Those with a Jacobian write the solution at that point into `integ.u`, where
OrdinaryDiffEq evaluates the Jacobian.
"""
linearize_observation!(linearization::AbstractTaylorLinearization, integ, x::Gaussian) =
    linearize_observation!(linearization, integ, x.μ)

function linearize_observation!(::ZeroJacobian, integ, m::AbstractVector)
    (; f, p, t, dt) = integ
    (; measurement_model, linearization_cache, E1, E2, R) = integ.cache
    (; r, H) = linearization_cache

    measurement_model(r, m, p, t + dt)
    integ.stats.nf += 1

    if f isa DynamicalODEFunction
        @assert f.mass_matrix === I
        copy!(H, E2)
    elseif f.mass_matrix === I
        copy!(H, E1)
    elseif f.mass_matrix isa UniformScaling
        rmul!(copy!(H, E1), f.mass_matrix.λ)
    else
        _matmul!(H, f.mass_matrix, E1)
    end
    return LinearizedObservation(m, r, H, R)
end

function linearize_observation!(
    linearization::Union{FullJacobian,DiagonalJacobian}, integ, m::AbstractVector,
)
    (; d, SolProj, linearization_cache) = integ.cache
    obs = linearize_observation!(ZeroJacobian(), integ, m)
    write_into_solution!(integ.u, m, integ.f; cache=integ.cache)
    J = linearization_cache.J
    OrdinaryDiffEqDifferentiation.calc_J!(J, integ, integ.cache, true)
    _matmul!(obs.H, jacobian_approximation(linearization, J, d), SolProj, -1.0, 1.0)
    return obs
end

# For second-order ODEs, `J` is the Jacobian of the first-order system in `(u', u)`, and
# its first `d` rows are the Jacobian of `f`
jacobian_approximation(::FullJacobian, J, d) = size(J, 2) != d ? view(J, 1:d, :) : J
function jacobian_approximation(::DiagonalJacobian, J, d)
    if size(J, 2) == d
        return Diagonal(J)
    else
        topleft = view(J, 1:d, 1:d)
        topright = view(J, 1:d, (d+1):2d)
        return BlocksOfDiagonals([[topleft[i, i] topright[i, i];] for i in 1:d])
    end
end
