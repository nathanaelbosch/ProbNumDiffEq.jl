"""
    AbstractLinearization

How an [`ODEFilter`](@ref) linearizes the observation of the ODE in each step.

A linearization `L <: AbstractLinearization` has methods for
- `linearize_ode_observation!(::L, integ, x::Gaussian)`, at the distribution `x`;
- `make_linearization_cache(::L, C, u, f)`, the buffers it writes into;
- `needs_jacobian(::L)`;
- optionally [`linearize_and_update!`](@ref), if the linearization depends on the update.
"""
abstract type AbstractLinearization end

"""
    AbstractTaylorLinearization <: AbstractLinearization

A linearization by a Taylor expansion of the ODE vector field at a point. At a `Gaussian`,
it linearizes at the mean. Its subtypes implement
`linearize_ode_observation!(::L, integ, m::AbstractVector)` at a point `m`, instead of the
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

"""
    make_linearization_cache(linearization, C::CovarianceStructure, u, f)

The buffers that [`linearize_ode_observation!`](@ref) writes into, for the covariance
structure `C`, the ODE state `u` and the ODE function `f`. Built once per solve, and stored
in `integ.cache.linearization_cache`.
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
    linearize_ode_observation!(linearization, integ, x::Gaussian)
    linearize_ode_observation!(::AbstractTaylorLinearization, integ, m::AbstractVector)

Linearize the observation of the ODE at the end of the current step at the distribution
`x`, and return it as a [`LinearizedObservation`](@ref). The observation is ``h(x) = 0``,
with ``h(x) = M E_1 x - f(E_0 x, t)``, or ``h(x) = E_2 x - f(E_1 x, E_0 x, t)`` for
second-order ODEs. A Taylor linearization linearizes at the mean of `x`, or at any point
`m`. Those with a Jacobian write the solution at that point into `integ.u`, where
OrdinaryDiffEq evaluates the Jacobian.
"""
linearize_ode_observation!(linearization::AbstractTaylorLinearization, integ, x::Gaussian) =
    linearize_ode_observation!(linearization, integ, x.μ)

function linearize_ode_observation!(::ZeroJacobian, integ, m::AbstractVector)
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

function linearize_ode_observation!(
    linearization::Union{FullJacobian,DiagonalJacobian}, integ, m::AbstractVector,
)
    (; d, SolProj, linearization_cache) = integ.cache
    obs = linearize_ode_observation!(ZeroJacobian(), integ, m)
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

"""
    Iterated(linearization; maxiters=20, tol=1e-2)

Linearize the ODE at the posterior mean that `linearization` gives, found by iteration:
starting from the predicted mean, linearize with `linearization`, update the mean, and repeat
until the update changes the solution by less than `tol` in units of `abstol` and `reltol`,
for at most `maxiters` iterations. The covariance comes from the last linearization. With
adaptive steps, a step whose iteration does not converge is rejected.

With `ZeroJacobian()`, `DiagonalJacobian()` or `FullJacobian()`, this is the iterated
extended Kalman filter (IEKF) [Bosch (2026)](@cite bosch26stable).

# Examples
```julia-repl
julia> solve(prob, ODEFilter(linearization=Iterated(DiagonalJacobian())))
```
"""
struct Iterated{L<:AbstractTaylorLinearization,T} <: AbstractLinearization
    linearization::L
    maxiters::Int
    tol::T
    function Iterated(linearization::AbstractTaylorLinearization; maxiters=20, tol=1e-2)
        maxiters >= 1 || throw(ArgumentError("`maxiters` must be at least 1."))
        return new{typeof(linearization),typeof(tol)}(linearization, maxiters, tol)
    end
end
function Base.show(io::IO, it::Iterated)
    print(io, "Iterated(", it.linearization)
    it.maxiters == 20 || print(io, ", maxiters=", it.maxiters)
    it.tol == 1e-2 || print(io, ", tol=", it.tol)
    print(io, ")")
end

needs_jacobian(it::Iterated) = needs_jacobian(it.linearization)

# The Taylor linearization that a linearization is based on
base_linearization(linearization::AbstractLinearization) = linearization
base_linearization(it::Iterated) = it.linearization

# The buffers of the wrapped linearization, the linearization points, and two solutions for
# the stopping rule
function make_linearization_cache(it::Iterated, C::CovarianceStructure{T}, u, f) where {T}
    return (; make_linearization_cache(it.linearization, C, u, f)...,
        point=similar(Array{T}, C.d * (C.q + 1)), u_new=similar(u), u_old=similar(u))
end

# The first linearization, at the predicted mean, is the wrapped one's; the iteration starts there
linearize_ode_observation!(it::Iterated, integ, x) =
    linearize_ode_observation!(it.linearization, integ, x)

"""
    linearize_and_update!(linearization, integ, x_filt, x_pred, obs)

Condition the prediction `x_pred` on the observation of the ODE, linearized with
`linearization`, and write the result into `x_filt`. `obs` is the linearization at the
predicted mean. Return what [`update!`](@ref) returns, or, if the step is rejected,
leave `integ.force_stepfail` set.

By default, this is the [`update!`](@ref) with `obs`. [`Iterated`](@ref) linearizes
again at the updated mean until it converges, and updates the covariance once.
"""
linearize_and_update!(::AbstractLinearization, integ, x_filt, x_pred, obs) =
    update!(x_filt, x_pred, obs; cache=integ.cache)
function linearize_and_update!(it::Iterated, integ, x_filt, x_pred, obs)
    relinearize(m) = linearize_ode_observation!(it.linearization, integ, m)
    small_change(μ, m) = scaled_change(integ, μ, m) < it.tol
    r = iterated_update_mean!(x_filt, x_pred, obs, relinearize, small_change;
        it.maxiters, m_cache=integ.cache.linearization_cache.point, cache=integ.cache)
    integ.stats.nnonliniter += r.iterations
    if !r.converged
        integ.stats.nnonlinconvfail += 1
        if integ.opts.adaptive
            integ.force_stepfail = true
            return nothing
        end
    end
    (; K, B, loglikelihood, ztSinvz, S, z) = r.upd
    update_cov!(x_filt, x_pred, r.obs, K, B; cache=integ.cache)
    return (; loglikelihood, ztSinvz, S, z)
end

# The norm of the change of the solution from the state mean `m` to `μ`, in units of `abstol`
# and `reltol` relative to the solution at `m`
function scaled_change(integ, μ, m)
    (; u_new, u_old) = integ.cache.linearization_cache
    t = integ.t + integ.dt
    write_into_solution!(u_new, μ, integ.f; cache=integ.cache)
    write_into_solution!(u_old, m, integ.f; cache=integ.cache)
    u_new .-= u_old
    DiffEqBase.calculate_residuals!(u_new, u_new, u_old, u_old, integ.opts.abstol,
        integ.opts.reltol, integ.opts.internalnorm, t)
    return integ.opts.internalnorm(u_new, t)
end
