function manifold_update!(x, h, maxiters=1, check=false)
    H = ForwardDiff.gradient(h, x.μ)
    result = DiffResults.GradientResult(x.μ)
    result = ForwardDiff.gradient!(result, h, x.μ)
    z = DiffResults.value(result)
    H = DiffResults.gradient(result)
    @assert H isa AbstractVector
    _H = reshape(H, 1, length(H))
    meas = Gaussian([z], X_A_Xt(x.Σ, _H))

    if iszero(meas.μ) return end
    update!(x, x, meas, _H)
end
