abstract type InitializationScheme end
struct TaylorModeInit <: InitializationScheme end
Base.@kwdef struct ClassicSolverInit{ALG} <: InitializationScheme
    alg::ALG = Tsit5()
    init_on_du::Bool = false
end


function initial_update!(integ, cache)
    @unpack u, f, p, t = integ
    @unpack d, x, Proj = cache
    q = integ.alg.order

    @unpack ddu, du, x_tmp, x_tmp2, m_tmp, K1 = cache

    # Initialize on u0
    condition_on!(x, Proj(0), view(u, :), m_tmp, K1, x_tmp.Σ, x_tmp2.Σ.mat)

    # Initialize on du0
    if isinplace(f)
        f(du, u, p, t)
    else
        du .= f(u, p, t)
    end
    condition_on!(x, Proj(1), view(du, :), m_tmp, K1, x_tmp.Σ, x_tmp2.Σ.mat)

    if q < 2
        return
    end

    # Use a jac or autodiff to initialize on ddu0
    if integ.alg.initialization isa TaylorModeInit || integ.alg.initialization.init_on_du
        if isinplace(f)
            dfdt = copy(u)
            ForwardDiff.derivative!(dfdt, (du, t) -> f(du, u, p, t), du, t)

            if !isnothing(f.jac)
                f.jac(ddu, u, p, t)
            else
                ForwardDiff.jacobian!(ddu, (du, u) -> f(du, u, p, t), du, u)
            end
        else
            dfdt = ForwardDiff.derivative((t) -> f(u, p, t), t)
            if !isnothing(f.jac)
                ddu .= f.jac(du, u, p, t)
            else
                ddu .= ForwardDiff.jacobian(u -> f(u, p, t), u)
            end
        end
        ddfddu = ddu * view(du, :) + view(dfdt, :)
        condition_on!(x, Proj(2), ddfddu, m_tmp, K1, x_tmp.Σ, x_tmp2.Σ.mat)
        if q < 3
            return
        end
    end

    # Compute the other parts with classic solvers

    initialize_higher_orders!(integ, cache, integ.alg.initialization)
end



########################################################################
# Some utilities below
"""Quick and dirty wrapper to make OOP functions IIP"""
function oop_to_iip(f)
    function f!(du, u, p, t)
        du .= f(u, p, t)
        return nothing
    end
    return f!
end

"""Basically an Kalman update"""
function condition_on!(
    x::SRGaussian,
    H::AbstractMatrix,
    data::AbstractVector,
    meascache,
    Kcache,
    covcache,
    Mcache,
)
    z, S = meascache

    _matmul!(z, H, x.μ)
    X_A_Xt!(S, x.Σ, H)
    @assert isdiag(S)
    S_diag = diag(S)
    if any(iszero.(S_diag)) # could happen with a singular mass-matrix
        S_diag .+= 1e-20
    end

    _matmul!(Kcache, x.Σ.mat, H')
    K = Kcache ./= S_diag'

    _matmul!(x.μ, K, data - z, 1.0, 1.0)
    # x.μ .+= K*(data - z)

    D = length(x.μ)
    mul!(Mcache, K, H, -1.0, 0.0)
    @inbounds @simd ivdep for i in 1:D
        Mcache[i, i] += 1
    end
    X_A_Xt!(covcache, x.Σ, Mcache)
    copy!(x.Σ, covcache)
    return nothing
end
