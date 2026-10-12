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
    linearize_observation!(linearization, integ, x)

Linearize the observation of the ODE at the end of the current step at `x`, and return it as
a [`LinearizedObservation`](@ref). The observation is ``h(x) = 0``, with
``h(x) = M E_1 x - f(E_0 x, t)``, or ``h(x) = E_2 x - f(E_1 x, E_0 x, t)`` for second-order
ODEs. `x` is a point or a `Gaussian`; linearizations at a point use the Gaussian's mean.
Linearizations with a Jacobian write the solution at that point into `integ.u`, where
OrdinaryDiffEq evaluates the Jacobian.
"""
linearize_observation!(linearization::AbstractLinearization, integ, x::Gaussian) =
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
