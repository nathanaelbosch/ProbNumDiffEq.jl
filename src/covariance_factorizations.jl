abstract type CovarianceFactorization end
struct KroneckerCovariance <: CovarianceFactorization end
struct DenseCovariance <: CovarianceFactorization end

function get_covariance_factorization(alg)
    if (
        alg isa EK0 &&
        !(
            alg.diffusionmodel isa DynamicMVDiffusion ||
            alg.diffusionmodel isa FixedMVDiffusion
        ) &&
        alg.prior isa IWP
    )
        return KroneckerCovariance()
    else
        return DenseCovariance()
    end
end

factorized_zeros(::KroneckerCovariance, elType, sizes...; d, q) = begin
    for s in sizes
        @assert s % d == 0
    end
    return IsoKroneckerProduct(d, zeros(elType, (s ÷ d for s in sizes)...))
end

factorized_zeros(::DenseCovariance, elType, sizes...; d, q) = zeros(elType, sizes...)

to_factorized_matrix(::DenseCovariance, M::AbstractMatrix) = Matrix(M)
to_factorized_matrix(::KroneckerCovariance, M::AbstractMatrix) = IsoKroneckerProduct(M) # probably errors
to_factorized_matrix(::KroneckerCovariance, M::IKP) = M
