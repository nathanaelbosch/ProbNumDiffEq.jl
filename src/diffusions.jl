abstract type AbstractDiffusion end
abstract type AbstractStaticDiffusion <: AbstractDiffusion end
abstract type AbstractDynamicDiffusion <: AbstractDiffusion end
isstatic(diffusion::AbstractStaticDiffusion) = true
isdynamic(diffusion::AbstractStaticDiffusion) = false
isstatic(diffusion::AbstractDynamicDiffusion) = false
isdynamic(diffusion::AbstractDynamicDiffusion) = true
initial_diffusion(diffusion::AbstractDiffusion, d, q, Eltype) = one(Eltype)

struct FixedDiffusion <: AbstractStaticDiffusion end
function estimate_diffusion(rule::FixedDiffusion, integ)
    @unpack d, measurement, m_tmp = integ.cache
    sol_diffusions = integ.sol.diffusions

    v, S = measurement.μ, measurement.Σ
    e, _ = m_tmp.μ, m_tmp.Σ.mat
    _S = copy!(m_tmp.Σ.mat, S.mat)

    if iszero(v)
        return zero(integ.cache.global_diffusion)
    end
    if iszero(S)
        return Inf
    end

    # diffusion_t = v' * inv(S) * v / d
    if _S isa Diagonal
        e .= v ./ _S.diag
    else
        S_chol = cholesky!(_S)
        ldiv!(e, S_chol, v)
    end
    diffusion_t = dot(v, e) / d

    if integ.success_iter == 0
        @assert length(sol_diffusions) == 0
        return diffusion_t, diffusion_t
    else
        @assert length(sol_diffusions) == integ.success_iter
        diffusion_prev = sol_diffusions[end]
        global_diffusion =
            diffusion_prev + (diffusion_t - diffusion_prev) / integ.success_iter
        return diffusion_t, global_diffusion
    end
end

"""Maximum a-posteriori Diffusion estimate when using an InverseGamma(1/2,1/2) prior

The mode of an InverseGamma(α,β) distribution is given by β/(α+1)
To compute this in an on-line basis from the previous Diffusion, we reverse the computation to
get the previous sum of residuals from Diffusion, and then modify that sum and compute the new
Diffusion.
"""
struct MAPFixedDiffusion <: AbstractStaticDiffusion end
function estimate_diffusion(rule::MAPFixedDiffusion, integ)
    @unpack d, measurement = integ.cache
    diffusions = integ.sol.diffusions

    N = integ.success_iter + 1
    v, S = measurement.μ, measurement.Σ
    res_t = v' * inv(S) * v / d

    α, β = 1 / 2, 1 / 2
    if integ.success_iter == 0
        @assert length(diffusions) == 0
        diffusion_t = (β + 1 / 2 * res_t) / (α + N * d / 2 + 1)
        return res_t, diffusion_t
    else
        @assert length(diffusions) == integ.success_iter
        diffusion_prev = diffusions[end]
        res_prev = (diffusion_prev * (α + (N - 1) * d / 2 + 1) - β) * 2
        res_sum_t = res_prev + res_t
        diffusion = (β + 1 / 2 * res_sum_t) / (α + N * d / 2 + 1)
        return res_t, diffusion
    end
end

struct DynamicDiffusion <: AbstractDynamicDiffusion end
function estimate_diffusion(kind::DynamicDiffusion, integ)
    @unpack d, R, H, Qh, measurement, m_tmp = integ.cache
    # @assert all(R .== 0) "The dynamic-diffusion assumes R==0!"
    z = measurement.μ
    e, HQH = m_tmp.μ, m_tmp.Σ
    X_A_Xt!(HQH, Qh, H)
    if HQH.mat isa Diagonal
        e .= z ./ HQH.mat.diag
        σ² = dot(e, z) / d
        return σ², σ²
    else
        C = cholesky!(HQH.mat)
        ldiv!(e, C, z)
        σ² = dot(z, e) / d
        return σ², σ²
    end
end

struct MVDynamicDiffusion <: AbstractDynamicDiffusion end
initial_diffusion(diffusion::MVDynamicDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
function estimate_diffusion(kind::MVDynamicDiffusion, integ)
    @unpack d, q, R, P, PI, E1 = integ.cache
    @unpack H, Qh, measurement, m_tmp = integ.cache
    z = measurement.μ

    # @assert all(R .== 0) "The dynamic-diffusion assumes R==0!"

    # Assert EK0
    @assert all(H .== E1)

    # More safety checks
    HQH = X_A_Xt!(m_tmp.Σ, Qh, H)
    @assert isdiag(HQH)
    @assert length(unique(diag(HQH))) == 1
    Q0_11 = diag(HQH)[1]

    Σ_ii = z .^ 2 ./ Q0_11
    Σ_ii .= max.(Σ_ii, eps(eltype(Σ_ii)))
    Σ = Diagonal(Σ_ii)

    Σ_out = kron(Σ, I(q + 1))
    @assert isdiag(Σ_out)
    # @info "MVDynamic diffusion" Σ Σ_out
    return Σ_out, Σ_out
end

struct MVFixedDiffusion <: AbstractStaticDiffusion end
initial_diffusion(diffusion::MVFixedDiffusion, d, q, Eltype) =
    kron(Diagonal(ones(Eltype, d)), Diagonal(ones(Eltype, q + 1)))
function estimate_diffusion(kind::MVFixedDiffusion, integ)
    @unpack d, q, R, P, PI, E1 = integ.cache
    @unpack measurement, H = integ.cache
    @unpack d, measurement = integ.cache
    diffusions = integ.sol.diffusions

    # Assert EK0
    @assert all(H .== E1)

    @unpack measurement = integ.cache
    v, S = measurement.μ, measurement.Σ

    # More safety checks
    @assert isdiag(S)
    # @assert length(unique(diag(S))) == 1
    S_11 = diag(S)[1]

    Σ_ii = v .^ 2 ./ S_11
    Σ = Diagonal(Σ_ii)
    Σ_out = kron(Σ, I(q + 1))
    @assert isdiag(Σ_out)
    # @info "MV-MLE-Diffusion" v S Σ Σ_out

    if integ.success_iter == 0
        @assert length(diffusions) == 0
        return Σ_out, Σ_out
    else
        @assert length(diffusions) == integ.success_iter
        diffusion_prev = diffusions[end]
        diffusion = diffusion_prev + (Σ_out - diffusion_prev) / integ.success_iter
        return Σ_out, diffusion
    end
end
