function projection(
    d::Integer,
    q::Integer,
    ::Type{elType}=typeof(1.0),
) where {elType}
    D = d * (q + 1)
    Proj(deriv) = begin
        P = zeros(elType, d, D)
        @simd ivdep for i in deriv*d+1:D+1:d*D
            @inbounds P[i] = 1
        end
        return P
    end
    return Proj
end
function projection(C::DenseCovariance{elType}) where {elType}
    projection(C.d, C.q, elType)
end

function projection(C::IsometricKroneckerCovariance{elType}) where {elType}
    Proj(deriv) = begin
        e_i = zeros(elType, C.q + 1, 1)
        if deriv <= C.q
            e_i[deriv+1] = 1
        end
        return IsometricKroneckerProduct(C.d, e_i')
    end
    return Proj
end

function solution_space_projection(C::CovarianceStructure, is_secondorder_ode)
    Proj = projection(C)
    if is_secondorder_ode
        return [Proj(1); Proj(0)]
    else
        return Proj(0)
    end
end

struct SecondOrderODESolutionProjector{T,FAC,M} <: AbstractMatrix{T}
    covariance_structure::FAC
    E0B::M
    E1B::M
end
function SecondOrderODESolutionProjector(C::IsometricKroneckerCovariance{T}) where {T}
    Proj = projection(C)
    E0B, E1B = Proj(0).B, Proj(1).B
    return SecondOrderODESolutionProjector{T,typeof(C),typeof(E0B)}(C, E0B, E1B)
end
function _gaussian_mul!(
    g_out::SRGaussian, M::SecondOrderODESolutionProjector, g_in::SRGaussian)
    @unpack d = M.covariance_structure
    E0 = IsometricKroneckerProduct(d, M.E0B)
    E1 = IsometricKroneckerProduct(d, M.E1B)
    _matmul!(view(g_out.μ, 1:d), E1, g_in.μ)
    _matmul!(view(g_out.μ, d+1:2d), E0, g_in.μ)
    _matmul!(g_out.Σ.R.A, g_in.Σ.R.B, [M.E1B; M.E0B]')
    return g_out
end
function Base.:*(M::SecondOrderODESolutionProjector, x::SRGaussian)
    @unpack d = M.covariance_structure
    E0 = IsometricKroneckerProduct(d, M.E0B)
    E1 = IsometricKroneckerProduct(d, M.E1B)
    μ = [E1 * x.μ; E0 * x.μ]
    Σ = PSDMatrix([x.Σ.R * E1' x.Σ.R * E0'])
    return Gaussian(μ, Σ)
end

function solution_space_projection(C::IsometricKroneckerCovariance, is_secondorder_ode)
    Proj = projection(C)
    if is_secondorder_ode
        return SecondOrderODESolutionProjector(C)
    else
        return Proj(0)
    end
end
