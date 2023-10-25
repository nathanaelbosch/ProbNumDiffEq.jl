abstract type CovarianceStructure end
struct IsometricKroneckerCovariance <: CovarianceStructure end
struct DenseCovariance <: CovarianceStructure end

function get_covariance_structure(alg)
    if (
        alg isa EK0 &&
        !(
            alg.diffusionmodel isa DynamicMVDiffusion ||
            alg.diffusionmodel isa FixedMVDiffusion
        ) &&
        alg.prior isa IWP
    )
        return IsometricKroneckerCovariance()
    else
        return DenseCovariance()
    end
end

factorized_zeros(::IsometricKroneckerCovariance, elType, sizes...; d, q) = begin
    for s in sizes
        @assert s % d == 0
    end
    return IsometricKroneckerProduct(d, Array{elType}(calloc, (s ÷ d for s in sizes)...))
end
factorized_similar(::IsometricKroneckerCovariance, elType, size1, size2; d, q) = begin
    for s in (size1, size2)
        @assert s % d == 0
    end
    return IsometricKroneckerProduct(d, similar(Matrix{elType}, size1 ÷ d, size2 ÷ d))
end

factorized_zeros(::DenseCovariance, elType, sizes...; d, q) = Array{elType}(calloc, sizes...)
factorized_similar(::DenseCovariance, elType, size1, size2; d, q) = similar(Matrix{elType}, size1, size2)

to_factorized_matrix(::DenseCovariance, M::AbstractMatrix) = Matrix(M)
to_factorized_matrix(::IsometricKroneckerCovariance, M::AbstractMatrix) =
    IsometricKroneckerProduct(M) # probably errors
to_factorized_matrix(::IsometricKroneckerCovariance, M::IsometricKroneckerProduct) = M
