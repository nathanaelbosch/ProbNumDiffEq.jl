abstract type InitializationScheme end
struct TaylorModeInit <: InitializationScheme end
struct RungeKuttaInit <: InitializationScheme end


########################################################################
# Some utilities below
"""Quick and dirty wrapper to make IIP functions OOP"""
function iip_to_oop(f!)
    function f(u, p, t)
        du = copy(u)
        f!(du, u, p, t)
        return du
    end
    return f
end
function oop_to_iip(f)
    function f!(du, u, p, t)
        du .= f(u, p, t)
        return nothing
    end
    return f!
end

"""Basically an Kalman update"""
function condition_on!(x::SRGaussian, H::AbstractMatrix, data::AbstractVector,
                       meascache, Kcache, Kcache2, covcache, Mcache)
    z, S = meascache

    _matmul!(z, H, x.μ)
    X_A_Xt!(S, x.Σ, H)
    @assert isdiag(S)

    _matmul!(Kcache, x.Σ.mat, H')
    Kcache2 .= Kcache ./ diag(S)'
    K = Kcache2

    _matmul!(x.μ, K, data - z, 1.0, 1.0)
    # x.μ .+= K*(data - z)

    D = length(x.μ)
    mul!(Mcache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        Mcache[i, i] += 1
    end
    X_A_Xt!(covcache, x.Σ, Mcache)
    copy!(x.Σ, covcache)
    nothing
end
