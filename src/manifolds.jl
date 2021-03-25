function manifold_update!(x, h, maxiters=1)
    result = DiffResults.GradientResult(x.μ)
    result = ForwardDiff.gradient!(result, h, x.μ)
    z = DiffResults.value(result)
    if iszero(z) return end
    H = DiffResults.gradient(result)
    @assert H isa AbstractVector

    SL = H'x.Σ
    meas = Gaussian(z, SL*SL')

    update!(x, x, meas, H)
end
