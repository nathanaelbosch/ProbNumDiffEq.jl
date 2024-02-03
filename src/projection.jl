function projection(
    d::Integer,
    q::Integer,
    ::Type{elType}=typeof(1.0),
) where {elType}
    D = d * (q + 1)
    Proj(deriv) = begin
        P = zeros(elType, d, D)
        if deriv <= q
            @simd ivdep for i in deriv*d+1:D+1:d*D
                @inbounds P[i] = 1
            end
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

function solution_space_projection(C::IsometricKroneckerCovariance, is_secondorder_ode)
    Proj = projection(C)
    if is_secondorder_ode
        return KroneckerSecondOrderODESolutionProjector(C)
    else
        return Proj(0)
    end
end

struct KroneckerSecondOrderODESolutionProjector{T,FAC,M,M2} <: AbstractMatrix{T}
    covariance_structure::FAC
    E0::M
    E1::M
    SolProjB::M2
end
function KroneckerSecondOrderODESolutionProjector(
    C::IsometricKroneckerCovariance{T},
) where {T}
    Proj = projection(C)
    E0, E1 = Proj(0), Proj(1)
    SolProjB = [E1.B; E0.B]
    return KroneckerSecondOrderODESolutionProjector{
        T,typeof(C),typeof(E0),typeof(SolProjB),
    }(
        C, E0, E1, SolProjB,
    )
end
function _gaussian_mul!(
    g_out::SRGaussian, M::KroneckerSecondOrderODESolutionProjector, g_in::SRGaussian)
    d = M.covariance_structure.d
    _matmul!(view(g_out.μ, 1:d), M.E1, g_in.μ)
    _matmul!(view(g_out.μ, d+1:2d), M.E0, g_in.μ)
    _matmul!(g_out.Σ.R.A, g_in.Σ.R.B, M.SolProjB')
    return g_out
end
function Base.:*(M::KroneckerSecondOrderODESolutionProjector, x::SRGaussian)
    μ = [M.E1 * x.μ; M.E0 * x.μ]
    Σ = PSDMatrix([x.Σ.R * M.E1' x.Σ.R * M.E0'])
    return Gaussian(μ, Σ)
end
