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
    return IsometricKroneckerProduct(d, zeros(elType, (s ÷ d for s in sizes)...))
end

factorized_zeros(::DenseCovariance, elType, sizes...; d, q) = zeros(elType, sizes...)

to_factorized_matrix(::DenseCovariance, M::AbstractMatrix) = Matrix(M)
to_factorized_matrix(::IsometricKroneckerCovariance, M::AbstractMatrix) =
    IsometricKroneckerProduct(M) # probably errors
to_factorized_matrix(::IsometricKroneckerCovariance, M::IsometricKroneckerProduct) = M
