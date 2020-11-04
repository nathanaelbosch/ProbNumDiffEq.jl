function assert_nonnegative_diagonal(cov)
    if !all(diag(cov) .>= 0)
        @error "Non-positive variances" cov diag(cov)
        error("The provided covariance has non-positive entries on the diagonal!")
    end
end
