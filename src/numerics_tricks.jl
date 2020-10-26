

function approx_diff!(A, B, C)
    @assert size(A) == size(B) == size(C)
    @assert eltype(A) == eltype(B) == eltype(C)
    # If B_ij ≈ C_ij, then A_ij = 0
    # But, only do this if the value in A is actually negative
    @simd for i in 1:length(A)
        @inbounds if B[i] ≈ C[i]
            A[i] = 0
        else
            A[i] = B[i] - C[i]
        end
    end
end


function fix_negative_variances(g::Gaussian, abstol::Real, reltol::Real)
    for i in 1:length(g.μ)
        if (g.Σ[i,i] < 0) && (sqrt(-g.Σ[i,i]) / (abstol + reltol * abs(g.μ[i]))) .< eps(eltype(g.Σ))
            # @info "fix_neg" g.Σ[i,i] g.μ[i] abstol reltol (abstol+reltol*abs(g.μ[i])) eps(eltype(g.Σ))*(abstol+reltol*abs(g.μ[i]))
            g.Σ[i,i] = eps(eltype(g.Σ))*(abstol+reltol*abs(g.μ[i]))
            g.Σ[i,i] = 0
        end
    end
end




function assert_nonnegative_diagonal(cov)
    if !all(diag(cov) .>= 0)
        @error "Non-positive variances" cov diag(cov)
        error("The provided covariance has non-positive entries on the diagonal!")
    end
end
