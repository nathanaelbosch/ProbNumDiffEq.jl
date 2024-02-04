abstract type CovarianceStructure{T} end
struct IsometricKroneckerCovariance{T} <: CovarianceStructure{T}
    d::Int64
    q::Int64
end
struct DenseCovariance{T} <: CovarianceStructure{T}
    d::Int64
    q::Int64
end

function get_covariance_structure(alg; elType, d, q)
    if (
        alg isa EK0 &&
        !(
            alg.diffusionmodel isa DynamicMVDiffusion ||
            alg.diffusionmodel isa FixedMVDiffusion
        ) &&
        alg.prior isa IWP
    )
        return IsometricKroneckerCovariance{elType}(d, q)
    else
        return DenseCovariance{elType}(d, q)
    end
end

factorized_zeros(C::IsometricKroneckerCovariance{T}, sizes...) where {T} = begin
    for s in sizes
        @assert s % C.d == 0
    end
    return IsometricKroneckerProduct(C.d, Array{T}(calloc, (s ÷ C.d for s in sizes)...))
end
factorized_similar(C::IsometricKroneckerCovariance{T}, size1, size2) where {T} = begin
    for s in (size1, size2)
        @assert s % C.d == 0
    end
    return IsometricKroneckerProduct(C.d, similar(Matrix{T}, size1 ÷ C.d, size2 ÷ C.d))
end

factorized_zeros(::DenseCovariance{T}, sizes...) where {T} =
    Array{T}(calloc, sizes...)
factorized_similar(::DenseCovariance{T}, size1, size2) where {T} =
    similar(Matrix{T}, size1, size2)

to_factorized_matrix(::DenseCovariance, M::AbstractMatrix) = Matrix(M)
to_factorized_matrix(::IsometricKroneckerCovariance, M::AbstractMatrix) =
    error("Cannot factorize Matrix")
to_factorized_matrix(::IsometricKroneckerCovariance, M::IsometricKroneckerProduct) = M
to_factorized_matrix(FAC::IsometricKroneckerCovariance, M::Diagonal{T, <:Fill{T, 1}}) where {T} =
	  IsometricKroneckerProduct(FAC.d, M.diag.value*Eye(FAC.q+1))

for FT in [:DenseCovariance, :IsometricKroneckerCovariance]
    @eval to_factorized_matrix(FAC::$FT, M::PSDMatrix) =
        PSDMatrix(to_factorized_matrix(FAC, M.R))
end
