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

"""To handle matrix-valued vector fields"""
function f_to_vector_valued(f::AbstractODEFunction{false}, u)
    u_template = copy(u)
    function new_f(u, p, t)
        du = f(reshape(u, size(u_template)), p, t)
        return du[:]
    end
    return new_f
end

"""Basically an Kalman update"""
function condition_on!(x::SRGaussian, H::AbstractMatrix, data::AbstractVector,
                       meascache, Kcache, Kcache2, covcache, Mcache)
    z, S = meascache

    _matmul!(z, H, x.μ)
    X_A_Xt!(S, x.Σ, H)
    @assert isdiag(S)
    S = Diagonal(S)

    _matmul!(Kcache, x.Σ.mat, H')
    Kcache2 .= Kcache ./ S.diag'
    K = Kcache2

    _matmul!(x.μ, K, data - z, 1, 1)
    # x.μ .+= K*(data - z)

    D = length(x.μ)
    _matmul!(Mcache, K, H, -1, 0)
    @inbounds @simd ivdep for i in 1:D
        Mcache[i, i] += 1
    end
    X_A_Xt!(covcache, x.Σ, Mcache)
    copy!(x.Σ, covcache)
    nothing
end
